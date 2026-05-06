use std::{
    env,
    io::{self, Write},
    path::{Path, PathBuf},
    process::Stdio,
};

use anyhow::Context;
use async_openai::traits::EventType;
use async_openai::{
    Client,
    types::{
        audio::{AudioResponseFormat, CreateTranscriptionRequestArgs},
        realtime::{
            RealtimeClientEventConversationItemCreate, RealtimeClientEventResponseCreate,
            RealtimeClientEventSessionUpdate, RealtimeConversationItem, RealtimeFunctionTool,
            RealtimeServerEvent, RealtimeSession, RealtimeTool, Session,
        },
    },
};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};
use futures_util::{
    SinkExt, StreamExt,
    stream::{SplitSink, SplitStream},
};
use serde_json::{Value, json};
use tokio::{
    fs,
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::TcpStream,
    process::Command,
};
use tokio_tungstenite::{
    MaybeTlsStream, WebSocketStream, connect_async,
    tungstenite::{client::IntoClientRequest, protocol::Message},
};

const REALTIME_MODEL: &str = "gpt-realtime";
const OUTPUT_SAMPLE_RATE: u32 = 24_000;
const OUTPUT_CHANNELS: u16 = 1;
const OUTPUT_BITS_PER_SAMPLE: u16 = 16;
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

type RealtimeWs = WebSocketStream<MaybeTlsStream<TcpStream>>;
type RealtimeWrite = SplitSink<RealtimeWs, Message>;
type RealtimeRead = SplitStream<RealtimeWs>;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let mut agent = RealtimeAgent::new().await?;
    agent.chat_loop().await
}

struct RealtimeAgent {
    write: RealtimeWrite,
    read: RealtimeRead,
}

impl RealtimeAgent {
    async fn new() -> Result<Self, anyhow::Error> {
        dotenvy::dotenv().ok();

        let api_key = env::var("OPENAI_API_KEY").context("请设置 OPENAI_API_KEY")?;
        let url = format!("wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}");

        let mut request = url.into_client_request()?;
        request
            .headers_mut()
            .insert("Authorization", format!("Bearer {api_key}").parse()?);

        let (ws_stream, _) = connect_async(request)
            .await
            .context("连接 Realtime WebSocket 失败")?;
        eprintln!("WebSocket handshake complete");

        let (write, read) = ws_stream.split();
        let mut agent = Self { write, read };
        agent.update_session().await?;

        Ok(agent)
    }

    async fn update_session(&mut self) -> anyhow::Result<()> {
        let session = RealtimeSession {
            instructions: Some(
                "你是一个强大的聊天机器人，请根据用户的提问进行答复，如果需要调用工具请直接调用，不知道请直接回复不清楚"
                    .to_string(),
            ),
            tools: Some(vec![RealtimeTool::Function(RealtimeFunctionTool {
                name: "calculator".to_string(),
                description: "一个可以计算数学表达式的计算器".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "需要计算的数学表达式，例如：'123+456+789'"
                        }
                    },
                    "required": ["expression"],
                    "additionalProperties": false
                }),
            })]),
            ..Default::default()
        };

        self.write
            .send(
                RealtimeClientEventSessionUpdate {
                    event_id: None,
                    session: Session::RealtimeSession(Box::new(session)),
                }
                .into(),
            )
            .await?;

        Ok(())
    }

    async fn chat_loop(&mut self) -> anyhow::Result<()> {
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

            if let Err(error) = self.process_query(&query).await {
                eprintln!("\nError: {error:#}");
            }
        }

        Ok(())
    }

    async fn process_query(&mut self, query: &str) -> anyhow::Result<String> {
        self.send_user_query(query).await?;

        print!("Assistant: ");
        io::stdout().flush()?;

        let mut answer = String::new();
        let mut audio_bytes = Vec::new();
        let mut waiting_for_tool_response = false;

        loop {
            let Some(message) = self.read.next().await else {
                anyhow::bail!("Realtime WebSocket 已关闭");
            };

            match message? {
                Message::Text(text) => {
                    let server_event: RealtimeServerEvent = serde_json::from_str(text.as_ref())?;
                    eprint!("{:40} | ", server_event.event_type());

                    match server_event {
                        RealtimeServerEvent::ResponseCreated(_) => {
                            if waiting_for_tool_response {
                                answer.clear();
                                waiting_for_tool_response = false;
                            }
                            audio_bytes.clear();
                        }
                        RealtimeServerEvent::ResponseOutputTextDelta(event) => {
                            answer.push_str(&event.delta);
                            print!("{}", event.delta);
                            io::stdout().flush()?;
                        }
                        RealtimeServerEvent::ResponseOutputAudioTranscriptDelta(event) => {
                            answer.push_str(&event.delta);
                            print!("{}", event.delta);
                            io::stdout().flush()?;
                        }
                        RealtimeServerEvent::ResponseOutputAudioDelta(event) => {
                            let decoded = BASE64_STANDARD.decode(event.delta)?;
                            eprint!("audio bytes: {}", decoded.len());
                            audio_bytes.extend_from_slice(&decoded);
                        }
                        RealtimeServerEvent::ResponseFunctionCallArgumentsDone(event) => {
                            answer.clear();
                            audio_bytes.clear();
                            self.run_tool_and_send_result(
                                &event.name,
                                &event.arguments,
                                &event.call_id,
                            )
                            .await?;
                            waiting_for_tool_response = true;
                        }
                        RealtimeServerEvent::ResponseDone(_) => {
                            if !audio_bytes.is_empty() {
                                play_audio(&audio_bytes).await?;
                                audio_bytes.clear();
                            }
                            eprintln!();

                            if waiting_for_tool_response {
                                continue;
                            }

                            println!();
                            return Ok(answer);
                        }
                        RealtimeServerEvent::ResponseOutputItemDone(event) => {
                            eprint!("{:?}", event.item);
                        }
                        RealtimeServerEvent::Error(error) => {
                            anyhow::bail!("Realtime 错误: {:?}", error.error);
                        }
                        _ => {}
                    }

                    eprintln!();
                }
                Message::Close(_) => anyhow::bail!("Realtime WebSocket 已关闭"),
                Message::Binary(_) => eprintln!("Binary"),
                Message::Frame(_) => eprintln!("Frame"),
                Message::Ping(_) => eprintln!("Ping"),
                Message::Pong(_) => eprintln!("Pong"),
            }
        }
    }

    async fn send_user_query(&mut self, query: &str) -> anyhow::Result<()> {
        let item = RealtimeConversationItem::try_from(json!({
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": query
                }
            ]
        }))?;

        let event: RealtimeClientEventConversationItemCreate = item.into();
        self.write.send(event.into()).await?;
        self.write
            .send(RealtimeClientEventResponseCreate::default().into())
            .await?;
        Ok(())
    }

    async fn run_tool_and_send_result(
        &mut self,
        tool_name: &str,
        arguments: &str,
        call_id: &str,
    ) -> anyhow::Result<()> {
        let tool_args: Value = serde_json::from_str(arguments)?;
        eprint!("Tool Call: {tool_name}, Parameters: {tool_args}");

        let result = match tool_name {
            "calculator" => calculator(tool_args["expression"].as_str().unwrap_or_default()),
            _ => json!({"error": format!("未知工具: {tool_name}")}).to_string(),
        };

        let item = RealtimeConversationItem::try_from(json!({
            "type": "function_call_output",
            "call_id": call_id,
            "output": result,
            "status": "completed"
        }))?;

        let event: RealtimeClientEventConversationItemCreate = item.into();
        self.write.send(event.into()).await?;
        self.write
            .send(RealtimeClientEventResponseCreate::default().into())
            .await?;
        Ok(())
    }
}

async fn speech_to_text() -> anyhow::Result<String> {
    let audio_path = std::env::temp_dir().join("0022_voice_broadcast_realtime_input.wav");

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
    let script_path = std::env::temp_dir().join("0022_voice_recorder.swift");
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

async fn play_audio(pcm_audio: &[u8]) -> anyhow::Result<()> {
    let audio_path = std::env::temp_dir().join("0022_voice_broadcast_realtime.wav");
    let wav_audio = wav_from_pcm16(pcm_audio);
    fs::write(&audio_path, wav_audio).await?;

    // 播放 Realtime 返回的 PCM 音频。Play PCM audio returned by Realtime.
    let status = Command::new("afplay")
        .arg(Path::new(&audio_path))
        .status()
        .await?;
    if !status.success() {
        anyhow::bail!("afplay 播放失败: {status}");
    }

    Ok(())
}

fn wav_from_pcm16(pcm_audio: &[u8]) -> Vec<u8> {
    let data_len = pcm_audio.len() as u32;
    let byte_rate =
        OUTPUT_SAMPLE_RATE * u32::from(OUTPUT_CHANNELS) * u32::from(OUTPUT_BITS_PER_SAMPLE) / 8;
    let block_align = OUTPUT_CHANNELS * OUTPUT_BITS_PER_SAMPLE / 8;

    let mut wav = Vec::with_capacity(44 + pcm_audio.len());
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16_u32.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes());
    wav.extend_from_slice(&OUTPUT_CHANNELS.to_le_bytes());
    wav.extend_from_slice(&OUTPUT_SAMPLE_RATE.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&OUTPUT_BITS_PER_SAMPLE.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    wav.extend_from_slice(pcm_audio);
    wav
}
