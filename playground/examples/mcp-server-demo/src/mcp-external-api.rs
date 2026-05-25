use anyhow::Result;
use reqwest::header::{ACCEPT, CONTENT_TYPE};
use rmcp::{
    ServerHandler,
    handler::server::wrapper::Parameters,
    schemars, tool, tool_handler, tool_router,
    transport::streamable_http_server::{
        StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
    },
};
use serde_json::json;

const DEFAULT_BIND_ADDRESS: &str = "127.0.0.1:9889";
const DEFAULT_LLMOPS_API: &str = "https://llmops.shortvar.com/api/openapi/chat";

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct AgentRequest {
    /// 用户需要提交的问题。/ User query sent to the external agent.
    query: String,
}

#[derive(Debug, serde::Deserialize)]
struct LlmopsStreamEvent {
    event: Option<String>,
    answer: Option<String>,
}

#[derive(Clone)]
struct ExternalApiServer;

#[tool_router]
impl ExternalApiServer {
    #[tool(description = "调用外部 Agent 回答问题，支持天气查询、网络搜索和当前时间等能力。")]
    async fn call_llmops_agent(
        &self,
        Parameters(AgentRequest { query }): Parameters<AgentRequest>,
    ) -> String {
        match request_llmops_agent(&query).await {
            Ok(answer) => answer,
            Err(err) => err,
        }
    }
}

#[tool_handler(
    name = "三方 API",
    version = "0.1.0",
    instructions = "提供一个 call_llmops_agent 工具，把 query 转发给外部 Agent 并返回最终答案。"
)]
impl ServerHandler for ExternalApiServer {}

#[tokio::main]
async fn main() -> Result<()> {
    let service: StreamableHttpService<ExternalApiServer, LocalSessionManager> =
        StreamableHttpService::new(
            || Ok(ExternalApiServer),
            Default::default(),
            StreamableHttpServerConfig::default(),
        );

    let app = axum::Router::new().nest_service("/mcp", service);
    let listener = tokio::net::TcpListener::bind(DEFAULT_BIND_ADDRESS).await?;

    eprintln!("External API MCP server listening on http://{DEFAULT_BIND_ADDRESS}/mcp");
    axum::serve(listener, app).await?;

    Ok(())
}

async fn request_llmops_agent(query: &str) -> Result<String, String> {
    let app_id = required_env("LLMOPS_APP_ID")?;
    let api_key = required_env("LLMOPS_API_KEY")?;
    let api_url = std::env::var("LLMOPS_API").unwrap_or_else(|_| DEFAULT_LLMOPS_API.to_string());

    let client = reqwest::Client::new();
    let response = client
        .post(api_url)
        .header(CONTENT_TYPE, "application/json")
        .header(ACCEPT, "application/json")
        .bearer_auth(api_key)
        .json(&json!({
            "query": query,
            "app_id": app_id,
            "stream": true,
        }))
        .send()
        .await
        .map_err(|err| format!("外部 API 请求失败: {err}"))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(format!("外部 API 请求失败: HTTP {status}, {body}"));
    }

    collect_agent_answer(response).await
}

async fn collect_agent_answer(mut response: reqwest::Response) -> Result<String, String> {
    let mut answer = String::new();
    let mut pending = String::new();

    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|err| format!("读取外部 API 流式响应失败: {err}"))?
    {
        pending.push_str(&String::from_utf8_lossy(&chunk));
        drain_complete_lines(&mut pending, &mut answer)?;
    }

    if !pending.trim().is_empty() {
        handle_sse_line(pending.trim(), &mut answer)?;
    }

    Ok(answer.trim().to_string())
}

fn drain_complete_lines(pending: &mut String, answer: &mut String) -> Result<(), String> {
    while let Some(line_end) = pending.find('\n') {
        let line = pending[..line_end].trim_end_matches('\r').to_string();
        pending.drain(..=line_end);
        handle_sse_line(&line, answer)?;
    }

    Ok(())
}

fn handle_sse_line(line: &str, answer: &mut String) -> Result<(), String> {
    let line = line.trim();
    if line.is_empty() {
        return Ok(());
    }

    let Some(data) = line.strip_prefix("data:") else {
        return Ok(());
    };

    let data = data.trim();
    if data.is_empty() || data == "[DONE]" {
        return Ok(());
    }

    let event: LlmopsStreamEvent = serde_json::from_str(data)
        .map_err(|err| format!("外部 API 流式数据解析失败: {err}; data={data}"))?;

    if event.event.as_deref() == Some("agent_message")
        && let Some(answer_part) = event.answer
    {
        answer.push_str(&answer_part);
    }

    Ok(())
}

fn required_env(name: &str) -> Result<String, String> {
    std::env::var(name)
        .map(|value| value.trim().to_string())
        .ok()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("缺少环境变量 {name}"))
}
