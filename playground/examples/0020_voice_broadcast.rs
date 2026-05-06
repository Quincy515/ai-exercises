use std::{
    collections::HashMap,
    io::{self, Write},
    path::PathBuf,
    process::Stdio,
};

use anyhow::{Context, anyhow};
use async_openai::types::audio::{
    AudioResponseFormat, CreateSpeechRequestArgs, CreateTranscriptionRequestArgs, SpeechModel,
    Voice,
};
use async_openai::types::chat::{
    ChatCompletionMessageToolCalls, ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessage, ChatCompletionRequestToolMessage,
    ChatCompletionRequestUserMessage, ChatCompletionResponseMessage, ChatCompletionTool,
    ChatCompletionToolChoiceOption, CreateChatCompletionRequestArgs, FunctionObjectArgs,
    ToolChoiceOptions,
};
use async_openai::{Client, config::OpenAIConfig};
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

    async fn process_query(&mut self, query: &str) -> Result<String, anyhow::Error> {
        self.messages
            .push(ChatCompletionRequestUserMessage::from(query).into());

        let response_message = self.create_chat_completion(None).await?;

        if let Some(tool_calls) = response_message
            .tool_calls
            .clone()
            .filter(|calls| !calls.is_empty())
        {
            self.messages.push(
                ChatCompletionRequestAssistantMessageArgs::default()
                    .tool_calls(tool_calls.clone())
                    .build()?
                    .into(),
            );

            for tool_call in tool_calls {
                let ChatCompletionMessageToolCalls::Function(tool_call) = tool_call else {
                    continue;
                };

                let tool_name = tool_call.function.name.as_str();
                let tool_args: Value = tool_call.function.arguments.parse()?;
                let expression = tool_args["expression"].as_str().unwrap_or_default();
                let function_to_call = self
                    .available_tools
                    .get(tool_name)
                    .ok_or_else(|| anyhow!("未知工具: {tool_name}"))?;
                let result = function_to_call(expression);
                println!("Tool Call: {tool_name}");
                println!("Tool [{tool_name}] Result: {result}");

                self.messages.push(
                    ChatCompletionRequestToolMessage {
                        content: result.into(),
                        tool_call_id: tool_call.id,
                    }
                    .into(),
                );
            }

            let second_response_message = self
                .create_chat_completion(Some(ChatCompletionToolChoiceOption::Mode(
                    ToolChoiceOptions::None,
                )))
                .await?;
            let content = second_response_message.content.clone().unwrap_or_default();
            self.messages
                .push(ChatCompletionRequestAssistantMessage::from(content.clone()).into());
            return Ok(format!("Assistant: {content}"));
        }

        let content = response_message.content.unwrap_or_default();
        self.messages
            .push(ChatCompletionRequestAssistantMessage::from(content.clone()).into());
        Ok(format!("Assistant: {content}"))
    }

    async fn create_chat_completion(
        &self,
        tool_choice: Option<ChatCompletionToolChoiceOption>,
    ) -> Result<ChatCompletionResponseMessage, anyhow::Error> {
        let mut request = CreateChatCompletionRequestArgs::default();
        request
            .model(MODEL)
            .messages(self.messages.clone())
            .tools(self.tool.clone());

        if let Some(tool_choice) = tool_choice {
            request.tool_choice(tool_choice);
        }

        let response = self.client.chat().create(request.build()?).await?;
        response
            .choices
            .first()
            .map(|choice| choice.message.clone())
            .ok_or_else(|| anyhow!("模型没有返回可用回复"))
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
                    println!("{answer}");
                    text_to_speech(&answer).await?;
                }
                Err(error) => eprintln!("\nError: {error:#}"),
            }
        }

        Ok(())
    }
}

async fn speech_to_text() -> anyhow::Result<String> {
    let audio_path = std::env::temp_dir().join("0020_voice_broadcast_input.wav");

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
    let script_path = std::env::temp_dir().join("0020_voice_recorder.swift");
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
        Ok(_) => json!({"error": "无效表达式, 错误信息: 结果不是有限数字"}).to_string(),
        Err(error) => json!({"error": format!("无效表达式, 错误信息: {error}")}).to_string(),
    }
}

async fn text_to_speech(text: &str) -> anyhow::Result<()> {
    let client = Client::new();
    let request = CreateSpeechRequestArgs::default()
        .model(SpeechModel::Tts1)
        .voice(Voice::Alloy)
        .input(text)
        .build()?;

    let audio_path: PathBuf = std::env::temp_dir().join("0020_voice_broadcast.mp3");
    let response = client.audio().speech().create(request).await?;
    response.save(&audio_path).await?;

    // 播放生成的语音文件。Play the generated speech file.
    let status = Command::new("afplay").arg(&audio_path).status().await?;
    if !status.success() {
        anyhow::bail!("afplay 播放失败: {status}");
    }

    Ok(())
}
