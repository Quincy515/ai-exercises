use std::{error::Error, io, time::Duration};

use serde_json::{Value, json};
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // 1. 定义 A2A 远程 Agent Card 基础 URL 地址
    let base_url = "http://localhost:9999";

    // 2. 创建一个 reqwest 客户端
    let http_client = reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()?;

    // 3. 获取 Agent 卡片信息
    let agent_card = http_client
        .get(format!("{base_url}/.well-known/agent-card.json"))
        .send()
        .await?
        .error_for_status()?
        .json::<Value>()
        .await?;
    println!(
        "Agent Card:\n{}",
        serde_json::to_string_pretty(&agent_card)?
    );

    // 4. 提取 Agent 卡片信息和请求端点
    // A2A V1 从 supportedInterfaces 中读取端点，不再使用顶层 url。
    let url = agent_card["supportedInterfaces"]
        .as_array()
        .and_then(|interfaces| {
            interfaces.iter().find(|interface| {
                interface["protocolBinding"] == "JSONRPC" && interface["protocolVersion"] == "1.0"
            })
        })
        .and_then(|interface| interface["url"].as_str())
        .ok_or_else(|| io::Error::other("Agent Card 中没有 A2A V1 JSONRPC 端点"))?;

    // 5. 构建发送消息请求体
    let request_body = json!({
        "id": Uuid::new_v4().to_string(),
        "jsonrpc": "2.0",
        "method": "SendMessage",
        "params": {
            "message": {
                "messageId": Uuid::new_v4().simple().to_string(),
                "role": "ROLE_USER",
                "parts": [
                    {"text": "帮我随机生成10个整数"}
                ]
            }
        }
    });

    // 6. 发起请求并打印响应内容
    let agent_response = http_client
        .post(url)
        .header("A2A-Version", "1.0")
        .json(&request_body)
        .send()
        .await?
        .error_for_status()?
        .json::<Value>()
        .await?;
    println!(
        "Agent Response:\n{}",
        serde_json::to_string_pretty(&agent_response)?
    );

    Ok(())
}
