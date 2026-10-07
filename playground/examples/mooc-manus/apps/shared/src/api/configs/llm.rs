//! 模型提供商接口：安全回显与只写密钥使用独立 DTO。

use crux_core::Request;
use crux_http::{
    command::{Http, RequestBuilder},
    protocol::HttpRequest,
};
use facet::Facet;
use serde::{Deserialize, Serialize};

use crate::api::endpoint;

const PATH: &str = "/api/app_configs/llm";

/// 服务端仅返回密钥是否已配置，结构可安全用于 ViewModel。
#[derive(Facet, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LlmConfig {
    pub base_url: Option<String>,
    pub model_name: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u64>,
    pub api_key_configured: bool,
}

/// 仅用于 HTTP 请求；密钥留空时省略，服务端保留已有密钥。
/// 此类型不导出到 FFI，也不实现 Debug，避免意外记录密钥。
#[derive(Serialize)]
pub struct LlmConfigRequest {
    pub base_url: Option<String>,
    #[serde(skip_serializing_if = "empty_api_key")]
    pub api_key: Option<String>,
    pub model_name: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u64>,
}

fn empty_api_key(value: &Option<String>) -> bool {
    value.as_deref().is_none_or(|key| key.trim().is_empty())
}

/// 构造模型提供商读取请求。
///
/// # Errors
/// 服务地址无效时返回固定错误信息。
pub fn get_llm_config<Effect, Event>(
    base_url: &str,
) -> Result<RequestBuilder<Effect, Event, LlmConfig>, String>
where
    Effect: From<Request<HttpRequest>> + Send + 'static,
    Event: Send + 'static,
{
    Ok(Http::get(endpoint(base_url, PATH)?).expect_json::<LlmConfig>())
}

/// 构造模型提供商更新请求。
///
/// # Errors
/// 服务地址无效或请求体序列化失败时返回固定错误信息。
pub fn update_llm_config<Effect, Event>(
    base_url: &str,
    config: &LlmConfigRequest,
) -> Result<RequestBuilder<Effect, Event, LlmConfig>, String>
where
    Effect: From<Request<HttpRequest>> + Send + 'static,
    Event: Send + 'static,
{
    Http::post(endpoint(base_url, PATH)?)
        .body_json(config)
        .map(|request| request.expect_json::<LlmConfig>())
        .map_err(|_| "配置请求编码失败，请重试。".to_string())
}
