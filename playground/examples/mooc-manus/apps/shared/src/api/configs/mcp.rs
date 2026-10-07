//! MCP 接口：安全列表与写确认 DTO；完整配置只进入请求体。

use std::collections::BTreeMap;

use crux_core::Request;
use crux_http::{
    Url,
    command::{Http, RequestBuilder},
    protocol::HttpRequest,
};
use facet::Facet;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

use crate::api::endpoint;

const PATH: &str = "/api/app_configs/mcp-servers";

#[derive(Facet, Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct McpServer {
    pub server_name: String,
    pub enabled: bool,
    #[serde(deserialize_with = "deserialize_transport")]
    pub transport: String,
    pub tools: Vec<String>,
}

#[derive(Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct McpServerList {
    pub mcp_servers: Vec<McpServer>,
}

/// 写接口返回完整配置；反序列化时仅保留确认所需的公开字段。
#[derive(Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct McpServerAck {
    #[serde(deserialize_with = "deserialize_transport")]
    pub transport: String,
    pub enabled: bool,
}

#[derive(Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct McpConfigAck {
    #[serde(rename = "mcpServers")]
    pub mcp_servers: BTreeMap<String, McpServerAck>,
}

/// 临时私有写入数据，不进入 FFI 类型或 Debug。
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct McpConfigRequest {
    #[serde(rename = "mcpServers")]
    pub mcp_servers: BTreeMap<String, Value>,
}

#[derive(Serialize)]
struct McpServerEnabledRequest {
    enabled: bool,
}

fn deserialize_transport<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let value = String::deserialize(deserializer)?;
    if matches!(value.as_str(), "stdio" | "streamable_http") {
        Ok(value)
    } else {
        Err(serde::de::Error::custom("unsupported MCP transport"))
    }
}

/// # Errors
/// 服务地址无效时返回固定错误信息。
pub fn get_mcp_servers<Effect, Event>(
    base_url: &str,
) -> Result<RequestBuilder<Effect, Event, McpServerList>, String>
where
    Effect: From<Request<HttpRequest>> + Send + 'static,
    Event: Send + 'static,
{
    Ok(Http::get(endpoint(base_url, PATH)?).expect_json::<McpServerList>())
}

/// # Errors
/// 服务地址无效或 JSON 编码失败时返回固定错误信息。
pub fn create_mcp_servers<Effect, Event>(
    base_url: &str,
    config: &McpConfigRequest,
) -> Result<RequestBuilder<Effect, Event, McpConfigAck>, String>
where
    Effect: From<Request<HttpRequest>> + Send + 'static,
    Event: Send + 'static,
{
    Http::post(endpoint(base_url, PATH)?)
        .body_json(config)
        .map(|request| request.expect_json::<McpConfigAck>())
        .map_err(|_| "配置请求编码失败，请重试。".to_string())
}

/// # Errors
/// 服务地址、资源名称无效或 JSON 编码失败时返回固定错误信息。
pub fn set_mcp_server_enabled<Effect, Event>(
    base_url: &str,
    server_name: &str,
    enabled: bool,
) -> Result<RequestBuilder<Effect, Event, McpConfigAck>, String>
where
    Effect: From<Request<HttpRequest>> + Send + 'static,
    Event: Send + 'static,
{
    Http::post(server_endpoint(base_url, server_name, "enabled")?)
        .body_json(&McpServerEnabledRequest { enabled })
        .map(|request| request.expect_json::<McpConfigAck>())
        .map_err(|_| "配置请求编码失败，请重试。".to_string())
}

/// # Errors
/// 服务地址或资源名称无效时返回固定错误信息。
pub fn delete_mcp_server<Effect, Event>(
    base_url: &str,
    server_name: &str,
) -> Result<RequestBuilder<Effect, Event, McpConfigAck>, String>
where
    Effect: From<Request<HttpRequest>> + Send + 'static,
    Event: Send + 'static,
{
    Ok(Http::post(server_endpoint(base_url, server_name, "delete")?).expect_json::<McpConfigAck>())
}

pub(crate) fn valid_server_name(name: &str) -> bool {
    !name.trim().is_empty() && !matches!(name, "." | "..") && !name.chars().any(char::is_control)
}

fn server_endpoint(base_url: &str, name: &str, action: &str) -> Result<Url, String> {
    if !valid_server_name(name) {
        return Err("MCP 服务名称无效，请检查名称后重试。".to_string());
    }
    let mut url = endpoint(base_url, PATH)?;
    url.path_segments_mut()
        .map_err(|()| "接口地址无效，请检查 API 路径。".to_string())?
        .push(name)
        .push(action);
    Ok(url)
}

#[cfg(test)]
mod tests {
    use super::{McpConfigAck, server_endpoint};

    #[test]
    fn encodes_names_as_one_path_segment() {
        assert_eq!(
            server_endpoint(
                "https://api.example.test",
                "mcp/../x?key=1#片段%2f",
                "delete"
            )
            .unwrap()
            .as_str(),
            "https://api.example.test/api/app_configs/mcp-servers/mcp%2F..%2Fx%3Fkey=1%23%E7%89%87%E6%AE%B5%252f/delete"
        );
        for name in ["", " ", ".", "..", "x\ny"] {
            assert!(server_endpoint("https://api.example.test", name, "delete").is_err());
        }
    }

    #[test]
    fn decodes_only_public_write_confirmation_fields() {
        let response = r#"{"mcpServers":{"demo":{"transport":"stdio","enabled":true,"env":{"TOKEN":"SECRET"},"headers":{"Authorization":"SECRET"},"args":["SECRET"],"command":"SECRET","url":"SECRET"}}}"#;
        let ack: McpConfigAck = serde_json::from_str(response).unwrap();
        assert_eq!(ack.mcp_servers["demo"].transport, "stdio");
        assert!(ack.mcp_servers["demo"].enabled);
        assert!(!format!("{ack:?}").contains("SECRET"));
        assert!(serde_json::from_str::<McpConfigAck>("null").is_err());
        assert!(
            serde_json::from_str::<McpConfigAck>(
                r#"{"mcpServers":{"demo":{"transport":"sse","enabled":true}}}"#
            )
            .is_err()
        );
    }
}
