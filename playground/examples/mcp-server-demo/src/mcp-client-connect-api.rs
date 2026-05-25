use anyhow::{Context, Result, bail};
use rmcp::{
    ServiceExt,
    model::{CallToolRequestParams, ClientInfo},
    object,
    transport::{
        StreamableHttpClientTransport, streamable_http_client::StreamableHttpClientTransportConfig,
    },
};

const DEFAULT_SERVER_URL: &str = "https://qianfan.baidubce.com/v2/ai_search/mcp";
const DEFAULT_QUERY: &str = "2025年广州马拉松";
const DEFAULT_TOOL_NAME: &str = "chatCompletions";

#[tokio::main]
async fn main() -> Result<()> {
    let config = ClientConfig::from_args()?;

    // 1. 创建 Streamable HTTP transport。/ Create the Streamable HTTP transport.
    let transport = streamable_http_transport(&config)?;

    // 2. 初始化 MCP Client 会话。/ Initialize the MCP client session.
    let client = ClientInfo::default()
        .serve(transport)
        .await
        .with_context(|| format!("连接 MCP Server 失败: {}", config.server_url))?;

    // 3. 获取工具列表并输出。/ List available tools and print their names.
    let tools = client.list_all_tools().await.context("获取工具列表失败")?;
    let tool_names = tools
        .iter()
        .map(|tool| tool.name.as_ref())
        .collect::<Vec<_>>();
    println!("工具列表: {tool_names:?}");

    // 4. 调用指定工具。/ Call the configured tool.
    let result = client
        .call_tool(
            CallToolRequestParams::new(config.tool_name.clone())
                .with_arguments(object!({ "query": config.query })),
        )
        .await
        .with_context(|| format!("调用 {} 工具失败", config.tool_name))?;

    // 5. 提取工具返回文本。/ Extract text content from the tool result.
    let result_text = result
        .content
        .iter()
        .filter_map(|content| content.raw.as_text())
        .map(|text| text.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    println!("工具调用结果: {result_text}");

    // 6. 关闭客户端。/ Close the client.
    client.cancel().await.context("关闭 MCP Client 失败")?;

    Ok(())
}

struct ClientConfig {
    server_url: String,
    query: String,
    tool_name: String,
    token: Option<String>,
}

impl ClientConfig {
    fn from_args() -> Result<Self> {
        let mut args = std::env::args().skip(1);
        let server_url = args
            .next()
            .or_else(|| non_empty_env("MCP_CONNECT_API_URL"))
            .or_else(|| non_empty_env("BAIDU_AI_SEARCH_API"))
            .unwrap_or_else(|| DEFAULT_SERVER_URL.to_string());
        let query = args
            .next()
            .or_else(|| non_empty_env("MCP_CONNECT_API_QUERY"))
            .unwrap_or_else(|| DEFAULT_QUERY.to_string());
        let tool_name = args
            .next()
            .or_else(|| non_empty_env("MCP_CONNECT_API_TOOL"))
            .unwrap_or_else(|| DEFAULT_TOOL_NAME.to_string());
        let token = non_empty_env("MCP_CONNECT_API_TOKEN")
            .or_else(|| non_empty_env("BAIDU_AI_SEARCH_API_KEY"));

        if server_url == DEFAULT_SERVER_URL && token.is_none() {
            bail!("缺少环境变量 MCP_CONNECT_API_TOKEN 或 BAIDU_AI_SEARCH_API_KEY");
        }

        Ok(Self {
            server_url,
            query,
            tool_name,
            token,
        })
    }
}

fn streamable_http_transport(
    config: &ClientConfig,
) -> Result<StreamableHttpClientTransport<reqwest::Client>> {
    let mut transport_config =
        StreamableHttpClientTransportConfig::with_uri(config.server_url.clone())
            .reinit_on_expired_session(true);

    if let Some(token) = &config.token {
        transport_config = transport_config.auth_header(bearer_token_value(token)?);
    }

    Ok(StreamableHttpClientTransport::from_config(transport_config))
}

fn bearer_token_value(token: &str) -> Result<String> {
    let token = token.trim();
    let token = token
        .strip_prefix("Bearer ")
        .or_else(|| token.strip_prefix("bearer "))
        .unwrap_or(token)
        .trim();

    if token.is_empty() {
        bail!("Authorization token 为空");
    }

    Ok(token.to_string())
}

fn non_empty_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}
