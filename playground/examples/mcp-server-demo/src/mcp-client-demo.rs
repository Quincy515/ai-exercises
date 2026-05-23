use anyhow::{Context, Result};
use rmcp::{
    ServiceExt,
    model::{CallToolRequestParams, ClientInfo},
    object,
    transport::StreamableHttpClientTransport,
};

const DEFAULT_SERVER_URL: &str = "http://127.0.0.1:8000/mcp";
const DEFAULT_EXPRESSION: &str = "564*34+12.4/455^2";

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let server_url = args
        .next()
        .unwrap_or_else(|| DEFAULT_SERVER_URL.to_string());
    let expression = args
        .next()
        .unwrap_or_else(|| DEFAULT_EXPRESSION.to_string());

    // 1. 初始化服务器的连接
    let transport = StreamableHttpClientTransport::from_uri(server_url.clone());

    // 2. 创建客户端
    let client = ClientInfo::default()
        .serve(transport)
        .await
        .with_context(|| format!("连接 MCP Server 失败: {server_url}"))?;

    // 3. 获取工具列表
    let tools = client.list_all_tools().await.context("获取工具列表失败")?;
    let tool_names = tools
        .iter()
        .map(|tool| tool.name.as_ref())
        .collect::<Vec<_>>();
    println!("工具列表: {tool_names:?}");

    // 4. 调用指定工具
    let result = client
        .call_tool(
            CallToolRequestParams::new("calculator")
                .with_arguments(object!({ "expression": expression })),
        )
        .await
        .context("调用 calculator 工具失败")?;

    // 5. 处理工具结果
    let result_text = result
        .content
        .iter()
        .filter_map(|content| content.raw.as_text())
        .map(|text| text.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    println!("工具结果: {result_text}");

    // 6. 关闭客户端
    client.cancel().await.context("关闭 MCP Client 失败")?;

    Ok(())
}
