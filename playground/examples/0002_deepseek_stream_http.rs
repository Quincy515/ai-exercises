//! DeepSeek 流式接口示例（手搓 SSE 解析）。
//!
//! SSE (Server-Sent Events) 协议三条核心规则：
//! 1. **事件边界是一个空行**（`\n\n`）。TCP 分包不保证按事件切分，
//!    所以必须把字节先缓冲到一个 buffer 里，凑齐 `\n\n` 才算一条完整事件。
//! 2. **一条事件可以包含多行**，每行是一个字段（如 `event:` / `data:` /
//!    `id:` / `retry:`），要按行独立解析、各自处理。
//! 3. **`data:` 字段可以在同一事件里出现多次**，规范要求用 `\n` 拼接
//!    成最终 payload。DeepSeek 每条事件只给一行 `data:`，所以这里简化处理。

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

    // 发起 POST 请求；`error_for_status` 让 4xx/5xx 在此处直接报错，
    // 避免把错误响应体当 SSE 流继续解析。
    let response = reqwest::Client::new()
        .post(api_url)
        .header("Authorization", format!("Bearer {api_key}"))
        .header("Content-Type", "application/json")
        .body(body.to_string())
        .send()
        .await?
        .error_for_status()?;

    // 规则 1：字节流到来的粒度由 TCP 决定，一个 chunk 里可能只有半条事件，
    // 也可能包含多条事件。用一个 String buffer 累积，遇到 `\n\n` 再切分。
    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    let mut stdout = std::io::stdout().lock();

    while let Some(chunk) = stream.next().await {
        buffer.push_str(&String::from_utf8_lossy(&chunk?));
        // while 而非 if：一个 chunk 可能一次性带来多条事件，全部消费掉。
        while let Some(pos) = buffer.find("\n\n") {
            let event: String = buffer.drain(..pos + 2).collect();
            if handle_event(&event, &mut stdout)? {
                writeln!(stdout)?;
                return Ok(());
            }
        }
    }
    writeln!(stdout)?;
    Ok(())
}
/// 解析一条完整 SSE 事件，输出增量内容。返回 `true` 表示收到 `[DONE]` 终止流。
fn handle_event(event: &str, out: &mut impl Write) -> anyhow::Result<bool> {
    // 规则 2：事件里可能有多行，逐行扫描；非 `data:` 开头的字段（如 `event:`、
    // 空行、注释行 `:heartbeat`）一律跳过。
    for line in event.lines() {
        let Some(data) = line.trim().strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        // DeepSeek 流的终止标记。
        if data == "[DONE]" {
            return Ok(true);
        }
        // 规则 3：此处假定 `data:` 一条事件里只有一行（DeepSeek 的惯例）。
        // 如果对接通用 SSE 服务，应把多行 `data:` 先拼接再 json 解析。
        let json: serde_json::Value = serde_json::from_str(data)?;
        if let Some(content) = json["choices"][0]["delta"]["content"].as_str() {
            write!(out, "{content}")?;
            out.flush()?;
        }
    }
    Ok(false)
}
