mod agent_executor;

use std::{error::Error, sync::Arc};

use a2a::{AgentCapabilities, AgentCard, AgentInterface, AgentSkill, TRANSPORT_PROTOCOL_JSONRPC};
use a2a_server::{DefaultRequestHandler, InMemoryTaskStore, StaticAgentCard};
use agent_executor::DeepSeekAgentExecutor;

const LISTEN_ADDRESS: &str = "0.0.0.0:9999";
const PUBLIC_BASE_URL: &str = "http://localhost:9999";

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // 1. 定义技能
    let skill = AgentSkill {
        id: "calculator".to_string(),
        name: "计算器".to_string(),
        description: "支持计算各种复杂数学公式".to_string(),
        tags: vec!["计算器".to_string()],
        examples: Some(vec!["445*34".to_string(), "211/34.2+12".to_string()]),
        input_modes: None,
        output_modes: None,
        security_requirements: None,
    };

    // 2. 定义 Agent 卡片
    // A2A V1 使用 supportedInterfaces 声明协议、版本和服务地址。
    let agent_card = AgentCard {
        name: "DeepSeek智能体".to_string(),
        description: "这是一个可以调用 DeepSeek 模型进行深度思考的智能体，在需要深度思考时可以使用"
            .to_string(),
        supported_interfaces: vec![AgentInterface::new(
            format!("{PUBLIC_BASE_URL}/jsonrpc"),
            TRANSPORT_PROTOCOL_JSONRPC,
        )],
        version: "1.0.0".to_string(),
        default_input_modes: vec!["text/plain".to_string()],
        default_output_modes: vec!["text/plain".to_string()],
        capabilities: AgentCapabilities {
            streaming: Some(false),
            extended_agent_card: Some(false),
            ..Default::default()
        },
        skills: vec![skill],
        provider: None,
        documentation_url: None,
        icon_url: None,
        security_schemes: None,
        security_requirements: None,
        signatures: None,
    };

    // 3. 使用 A2A 默认请求处理器（JSON-RPC）
    let request_handler = Arc::new(DefaultRequestHandler::new(
        DeepSeekAgentExecutor::new(),
        InMemoryTaskStore::new(),
    ));

    // 4. 创建并启动 A2A 服务器
    let card_producer = Arc::new(StaticAgentCard::new(agent_card));
    let app = axum::Router::new()
        .nest(
            "/jsonrpc",
            a2a_server::jsonrpc::jsonrpc_router(request_handler),
        )
        .merge(a2a_server::agent_card::agent_card_router(card_producer));

    println!("DeepSeek A2A Agent 已启动");
    println!("Agent Card: {PUBLIC_BASE_URL}/.well-known/agent-card.json");
    println!("JSON-RPC:   {PUBLIC_BASE_URL}/jsonrpc");

    let listener = tokio::net::TcpListener::bind(LISTEN_ADDRESS).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
