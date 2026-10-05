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

/// GET /api/app_configs/agent 直接返回此结构。
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
