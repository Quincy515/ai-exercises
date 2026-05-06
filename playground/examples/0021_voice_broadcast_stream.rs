use std::{
    collections::HashMap,
    io::{self, Write},
    path::PathBuf,
    process::Stdio,
};

use anyhow::{Context, anyhow};
use async_openai::traits::EventType;
use async_openai::types::audio::{
    AudioResponseFormat, CreateSpeechRequestArgs, CreateSpeechResponseStreamEvent,
    CreateTranscriptionRequestArgs, SpeechModel, StreamFormat, Voice,
};
use async_openai::types::chat::{
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCallChunk,
    ChatCompletionMessageToolCalls, ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessage, ChatCompletionRequestToolMessage,
    ChatCompletionRequestUserMessage, ChatCompletionResponseStream, ChatCompletionTool,
    ChatCompletionToolChoiceOption, CreateChatCompletionRequestArgs, FunctionCall,
    FunctionCallStream, FunctionObjectArgs, ToolChoiceOptions,
};
use async_openai::{Client, config::OpenAIConfig};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};
use futures_util::StreamExt;
use serde_json::{Value, json};
use tokio::{
    fs,
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::Command,
};

const MODEL: &str = "gpt-5.4-mini";
const RECORDER_SWIFT: &str = r#"
import AVFoundation
import Foundation

let outputPath = CommandLine.arguments[1]
let outputURL = URL(fileURLWithPath: outputPath)
let settings: [String: Any] = [
    AVFormatIDKey: Int(kAudioFormatLinearPCM),
    AVSampleRateKey: 16000,
    AVNumberOfChannelsKey: 1,
    AVLinearPCMBitDepthKey: 16,
    AVLinearPCMIsFloatKey: false,
    AVLinearPCMIsBigEndianKey: false
]

let recorder = try AVAudioRecorder(url: outputURL, settings: settings)
recorder.prepareToRecord()

guard recorder.record() else {
    fputs("record failed\n", stderr)
    exit(1)
}

_ = readLine()
recorder.stop()
"#;
type ToolFn = fn(&str) -> String;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let mut agent = ReActAgent::new()?;
    agent.chat_loop().await
}

struct ReActAgent {
    client: Client<OpenAIConfig>,
    messages: Vec<ChatCompletionRequestMessage>,
    tool: ChatCompletionTool,
    available_tools: HashMap<&'static str, ToolFn>,
}

impl ReActAgent {
    fn new() -> Result<Self, anyhow::Error> {
        dotenvy::dotenv().ok();

        let client = Client::new();
        let messages = vec![ChatCompletionRequestSystemMessage::from("你是一个强大的聊天机器人，请根据用户的提问进行答复，如果需要调用工具请直接调用，不知道请直接回复不清楚").into()];
        let tool = ChatCompletionTool {
            function: FunctionObjectArgs::default()
                .name("calculator")
                .description("一个可以计算数学表达式的计算器")
                .parameters(json!({
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "需要计算的数学表达式，例如：'123+456+789'"
                        }
                    },
                    "required": ["expression"],
                    "additionalProperties": false
                }))
                .strict(true)
                .build()?,
        };

        Ok(Self {
            client,
            messages,
            tool,
            available_tools: HashMap::from([("calculator", calculator as ToolFn)]),
        })
    }

    async fn process_query(&mut self, query: &str) -> anyhow::Result<String> {
        self.messages
            .push(ChatCompletionRequestUserMessage::from(query).into());

        print!("Assistant: ");
        io::stdout().flush()?;

        let mut response = self.create_chat_completion_stream(None).await?;
        let (content, tool_calls) = receive_chat_stream(&mut response).await?;

        if tool_calls.is_empty() {
            self.messages
                .push(ChatCompletionRequestAssistantMessage::from(content.clone()).into());
            println!();
            return Ok(content);
        }

        let mut assistant_message = ChatCompletionRequestAssistantMessageArgs::default();
        if !content.is_empty() {
            assistant_message.content(content);
        }
        assistant_message.tool_calls(tool_calls.clone());
        self.messages.push(assistant_message.build()?.into());

        for tool_call in tool_calls {
            let ChatCompletionMessageToolCalls::Function(tool_call) = tool_call else {
                continue;
            };

            let tool_name = tool_call.function.name.as_str();
            let tool_args: Value = tool_call.function.arguments.parse()?;
            println!("\nTool Call: {tool_name}");
            println!("Tool Parameters: {tool_args}");

            let expression = tool_args["expression"].as_str().unwrap_or_default();
            let function_to_call = self
                .available_tools
                .get(tool_name)
                .ok_or_else(|| anyhow!("未知工具: {tool_name}"))?;
            let result = function_to_call(expression);
            println!("Tool [{tool_name}] Result: {result}");

            self.messages.push(
                ChatCompletionRequestToolMessage {
                    content: result.into(),
                    tool_call_id: tool_call.id,
                }
                .into(),
            );
        }

        let mut second_response = self
            .create_chat_completion_stream(Some(ChatCompletionToolChoiceOption::Mode(
                ToolChoiceOptions::None,
            )))
            .await?;

        print!("Assistant: ");
        io::stdout().flush()?;

        let (final_content, _) = receive_chat_stream(&mut second_response).await?;
        self.messages
            .push(ChatCompletionRequestAssistantMessage::from(final_content.clone()).into());
        println!();

        Ok(final_content)
    }

    async fn create_chat_completion_stream(
        &self,
        tool_choice: Option<ChatCompletionToolChoiceOption>,
    ) -> Result<ChatCompletionResponseStream, anyhow::Error> {
        let mut request = CreateChatCompletionRequestArgs::default();
        request
            .model(MODEL)
            .messages(self.messages.clone())
            .tools(self.tool.clone())
            .stream(true);

        if let Some(tool_choice) = tool_choice {
            request.tool_choice(tool_choice);
        }

        Ok(self.client.chat().create_stream(request.build()?).await?)
    }

    async fn chat_loop(&mut self) -> Result<(), anyhow::Error> {
        loop {
            let query = match speech_to_text().await {
                Ok(query) => query,
                Err(error) => {
                    eprintln!("\nError: {error:#}");
                    break;
                }
            };

            let query = query.trim().to_string();
            println!("\nQuery: {query}");
            if query.is_empty() {
                continue;
            }
            if query == "退出" || query.eq_ignore_ascii_case("quit") {
                break;
            }

            match self.process_query(&query).await {
                Ok(answer) => {
                    text_to_speech_stream(&answer).await?;
                }
                Err(error) => eprintln!("\nError: {error:#}"),
            }
        }

        Ok(())
    }
}

async fn receive_chat_stream(
    response: &mut ChatCompletionResponseStream,
) -> anyhow::Result<(String, Vec<ChatCompletionMessageToolCalls>)> {
    let mut content = String::new();
    let mut tool_calls_object: HashMap<u32, ChatCompletionMessageToolCallChunk> = HashMap::new();

    while let Some(chunk_result) = response.next().await {
        let chunk = chunk_result?;
        let Some(choice) = chunk.choices.first() else {
            continue;
        };

        let delta = &choice.delta;
        if let Some(chunk_content) = delta.content.as_ref() {
            content.push_str(chunk_content);
            print!("{chunk_content}");
            io::stdout().flush()?;
        }

        if let Some(chunk_tool_calls) = &delta.tool_calls {
            for chunk_tool_call in chunk_tool_calls {
                let entry = tool_calls_object
                    .entry(chunk_tool_call.index)
                    .or_insert_with(|| ChatCompletionMessageToolCallChunk {
                        index: chunk_tool_call.index,
                        id: None,
                        r#type: None,
                        function: None,
                    });

                if entry.id.is_none() {
                    entry.id = chunk_tool_call.id.clone();
                }

                if entry.r#type.is_none() {
                    entry.r#type = chunk_tool_call.r#type.clone();
                }

                if let Some(function_delta) = &chunk_tool_call.function {
                    let function = entry.function.get_or_insert(FunctionCallStream {
                        name: None,
                        arguments: None,
                    });

                    if function.name.is_none() {
                        function.name = function_delta.name.clone();
                    }

                    if let Some(arguments_delta) = &function_delta.arguments {
                        function
                            .arguments
                            .get_or_insert_with(String::new)
                            .push_str(arguments_delta);
                    }
                }
            }
        }
    }

    let mut tool_call_chunks: Vec<_> = tool_calls_object.into_values().collect();
    tool_call_chunks.sort_by_key(|tool_call| tool_call.index);

    let tool_calls = tool_call_chunks
        .into_iter()
        .map(|tool_call| {
            let function = tool_call.function.ok_or_else(|| anyhow!("工具调用缺少"))?;

            Ok(ChatCompletionMessageToolCalls::Function(
                ChatCompletionMessageToolCall {
                    id: tool_call.id.ok_or_else(|| anyhow!("工具调用缺少id"))?,
                    function: FunctionCall {
                        name: function
                            .name
                            .ok_or_else(|| anyhow!("工具调用缺少 function.name"))?,
                        arguments: function.arguments.unwrap_or_default(),
                    },
                },
            ))
        })
        .collect::<anyhow::Result<_>>()?;

    Ok((content, tool_calls))
}

async fn speech_to_text() -> anyhow::Result<String> {
    let audio_path = std::env::temp_dir().join("0021_voice_broadcast_stream_input.wav");

    println!("按回车开始录音，再按一次回车停止录音...");
    wait_for_enter().await?;
    record_audio(&audio_path).await?;

    // 调用 OpenAI 语音转文本接口。Call OpenAI speech-to-text with the recorded file.
    let client = Client::new();
    let request = CreateTranscriptionRequestArgs::default()
        .file(audio_path)
        .model("whisper-1")
        .response_format(AudioResponseFormat::Text)
        .build()?;
    let response = client.audio().transcription().create_raw(request).await?;

    Ok(String::from_utf8_lossy(response.as_ref())
        .trim()
        .to_string())
}

async fn record_audio(audio_path: &PathBuf) -> anyhow::Result<()> {
    let script_path = std::env::temp_dir().join("0021_voice_recorder.swift");
    fs::write(&script_path, RECORDER_SWIFT).await?;

    let mut child = Command::new("swift")
        .arg(&script_path)
        .arg(audio_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .context("启动录音程序失败")?;

    println!("录音中... 再按回车停止");
    wait_for_enter().await?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(b"\n").await?;
    }

    let output = child.wait_with_output().await?;
    if !output.status.success() {
        let error = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("录音失败: {error}");
    }

    Ok(())
}

async fn wait_for_enter() -> anyhow::Result<()> {
    io::stdout().flush()?;
    let mut reader = BufReader::new(tokio::io::stdin());
    let mut line = Vec::new();
    if reader.read_until(b'\n', &mut line).await? == 0 {
        anyhow::bail!("输入结束");
    }
    Ok(())
}

fn calculator(expression: &str) -> String {
    match fasteval::ez_eval(expression, &mut fasteval::EmptyNamespace) {
        Ok(result) if result.is_finite() => json!({"result": result}).to_string(),
        Ok(_) => json!({"error": "无效表达式, 错误信息: 结果必须是有限数字"}).to_string(),
        Err(error) => json!({"error": format!("无效表达式, 错误信息: {error}")}).to_string(),
    }
}

async fn text_to_speech_stream(text: &str) -> anyhow::Result<()> {
    let client = Client::new();
    let request = CreateSpeechRequestArgs::default()
        .input(text)
        .voice(Voice::Alloy)
        .model(SpeechModel::Gpt4oMiniTts)
        .stream_format(StreamFormat::SSE)
        .build()?;

    let audio_path = std::env::temp_dir().join("0021_voice_broadcast_stream.mp3");
    let mut response = client.audio().speech().create_stream(request).await?;
    let mut file = fs::File::create(&audio_path).await?;

    while let Some(event_result) = response.next().await {
        match event_result? {
            CreateSpeechResponseStreamEvent::SpeechAudioDelta(delta) => {
                let decoded = BASE64_STANDARD.decode(&delta.audio)?;
                println!(
                    "[{}] audio base64-decoded size: {}",
                    delta.event_type(),
                    decoded.len()
                );
                file.write_all(&decoded).await?;
            }
            CreateSpeechResponseStreamEvent::SpeechAudioDone(done) => {
                println!("[{}] usage: {:?}", done.event_type(), done.usage);
            }
        }
    }

    file.flush().await?;
    drop(file);

    // 播放流式写完的语音文件。Play the speech file after streamed audio is written.
    let status = Command::new("afplay").arg(&audio_path).status().await?;
    if !status.success() {
        anyhow::bail!("afplay 播放失败: {status}");
    }

    Ok(())
}
