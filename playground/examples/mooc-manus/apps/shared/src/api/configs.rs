//! 设置接口：请求地址、响应类型和 HTTP 请求构造。

use crux_core::Request;
use crux_http::{
    command::{Http, RequestBuilder},
    protocol::HttpRequest,
};
use facet::Facet;
use serde::{Deserialize, Serialize};

use super::endpoint;

const PATH: &str = "/api/app_configs/agent";

/// Agent 配置读写接口直接返回此结构，更新时也使用这三个字段。
#[derive(Facet, Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct AgentConfig {
    pub max_iterations: i64,
    pub max_retries: i64,
    pub max_search_results: i64,
}

/// 构造 Agent 配置请求，由业务模块将响应映射为自己的事件。
///
/// # Errors
/// 服务地址格式无效或协议不受支持时返回错误信息。
pub fn get_agent_config<Effect, Event>(
    base_url: &str,
) -> Result<RequestBuilder<Effect, Event, AgentConfig>, String>
where
    Effect: From<Request<HttpRequest>> + Send + 'static,
    Event: Send + 'static,
{
    let url = endpoint(base_url, PATH)?;
    Ok(Http::get(url).expect_json::<AgentConfig>())
}

/// 更新 Agent 配置，由业务模块校验输入并处理响应。
///
/// # Errors
/// 服务地址无效或请求体序列化失败时返回错误信息。
pub fn update_agent_config<Effect, Event>(
    base_url: &str,
    config: &AgentConfig,
) -> Result<RequestBuilder<Effect, Event, AgentConfig>, String>
where
    Effect: From<Request<HttpRequest>> + Send + 'static,
    Event: Send + 'static,
{
    let url = endpoint(base_url, PATH)?;
    Http::post(url)
        .body_json(config)
        .map(|request| request.expect_json::<AgentConfig>())
        .map_err(|_| "配置请求编码失败，请重试。".to_string())
}
