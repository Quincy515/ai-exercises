use anyhow::Result;
use rmcp::{
    ServerHandler, ServiceExt, handler::server::wrapper::Parameters, schemars, tool, tool_handler,
    tool_router, transport::stdio,
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
    let service = CalculatorServer.serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}
