use anyhow::Result;
use rmcp::{
    ServerHandler,
    handler::server::wrapper::Parameters,
    schemars, tool, tool_handler, tool_router,
    transport::streamable_http_server::{
        StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
    },
};
use serde_json::json;

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct CalculatorRequest {
    /// 符合数学表达式语法的字符串，例如 `123 + 456 * 2`。/ Math expression string, for example `123 + 456 * 2`.
    expression: String,
}

#[derive(Clone)]
struct CalculatorServer;

#[tool_router]
impl CalculatorServer {
    #[tool(description = "一个数学计算器，用于计算传递的数学表达式")]
    async fn calculator(
        &self,
        Parameters(CalculatorRequest { expression }): Parameters<CalculatorRequest>,
    ) -> String {
        match fasteval::ez_eval(&expression, &mut fasteval::EmptyNamespace) {
            Ok(result) => json!({ "result": result }).to_string(),
            Err(err) => json!({ "result": format!("数学表达式计算出错: {err}") }).to_string(),
        }
    }
}

#[tool_handler(
    name = "calculator",
    version = "0.1.0",
    instructions = "提供一个 calculator 工具，输入 expression 字符串并返回 JSON 字符串。"
)]
impl ServerHandler for CalculatorServer {}

#[tokio::main]
async fn main() -> Result<()> {
    let service: StreamableHttpService<CalculatorServer, LocalSessionManager> =
        StreamableHttpService::new(
            || Ok(CalculatorServer),
            Default::default(),
            StreamableHttpServerConfig::default(),
        );

    let app = axum::Router::new().nest_service("/mcp", service);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:8000").await?;

    eprintln!("Calculator MCP server listening on http://127.0.0.1:8000/mcp");
    axum::serve(listener, app).await?;

    Ok(())
}
