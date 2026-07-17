use std::{error::Error, sync::Arc, time::Duration};

use a2a::{Message, Part, Role, SendMessageRequest, TRANSPORT_PROTOCOL_JSONRPC};
use a2a_client::{
    A2AClientFactory, agent_card::AgentCardResolver, jsonrpc::JsonRpcTransportFactory,
};

const BASE_URL: &str = "http://localhost:9999";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // 1. 定义 A2A 基础 URL 地址
    let base_url = BASE_URL;

    // 2. 创建一个 HTTP 客户端
    // Rust 的 reqwest::Client 可以复用连接，不需要 Python 的异步上下文管理器。
    let http_client = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()?;

    // 3. 创建一个 Agent 卡片解析器
    // A2A V1 会从 {base_url}/.well-known/agent-card.json 获取 Agent Card。
    let resolver = AgentCardResolver::new(Some(http_client.clone()));
    let card = resolver.resolve(base_url).await?;
    println!("Agent Card:\n{}", serde_json::to_string_pretty(&card)?);

    // 4. 创建一个 A2A 客户端
    // 客户端根据 Agent Card 中的 supportedInterfaces 找到 JSON-RPC 端点。
    let factory = A2AClientFactory::builder()
        .no_defaults()
        .register(Arc::new(JsonRpcTransportFactory::new(Some(http_client))))
        .preferred_bindings(vec![TRANSPORT_PROTOCOL_JSONRPC.to_string()])
        .build();
    let client = factory.create_from_card(&card).await?;

    // 5. 构建发送消息载体
    // Message::new 会自动生成 messageId；JSON-RPC 请求 ID 由 SDK 的 Transport 生成。
    let request = SendMessageRequest {
        message: Message::new(Role::User, vec![Part::text("帮我随机生成10个整数")]),
        configuration: None,
        metadata: None,
        tenant: None,
    };
    let response = client.send_message(&request).await?;

    // 6. 打印响应内容
    println!("Response:\n{}", serde_json::to_string_pretty(&response)?);

    Ok(())
}
