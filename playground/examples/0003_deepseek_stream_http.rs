//! DeepSeek 流式接口示例（原始 SSE 打印，不做 JSON 解析）。
//!
//! 直接把服务端返回的字节流打印到 stdout，方便观察 SSE 原文：
//! 每条事件以 `data: {...}` 开头，事件之间用空行（`\n\n`）分隔，
//! 流末尾是 `data: [DONE]`。

use futures_util::StreamExt;
use std::io::Write;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    dotenvy::dotenv().ok();
    let api_key = std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY must be set");
    let api_url = std::env::var("DEEPSEEK_API_URL")
        .unwrap_or_else(|_| "https://api.deepseek.com/v1/chat/completions".to_string());
    let body = serde_json::json!({
        "model": "deepseek-v4-flash",
        "messages": [
            { "role": "user", "content": "你好，你是？" }
        ],
        "stream": true
    });

    // `error_for_status` 让 4xx/5xx 直接报错，避免把错误响应体当 SSE 流处理。
    let response = reqwest::Client::new()
        .post(api_url)
        .header("Authorization", format!("Bearer {api_key}"))
        .header("Content-Type", "application/json")
        .body(body.to_string())
        .send()
        .await?
        .error_for_status()?;

    // 边收边打印：来多少字节打多少字节，保持 SSE 原始样貌。
    let mut stream = response.bytes_stream();
    let mut stdout = std::io::stdout().lock();
    while let Some(chunk) = stream.next().await {
        stdout.write_all(&chunk?)?;
        stdout.flush()?;
    }
    Ok(())
}
