//! A2A 配置接口：列表 DTO 与新增、启停、删除请求。

use crux_core::Request;
use crux_http::{
    Url,
    command::{Http, RequestBuilder},
    protocol::HttpRequest,
};
use facet::Facet;
use serde::{Deserialize, Serialize};

use crate::api::endpoint;

const PATH: &str = "/api/app_configs/a2a-servers";

#[derive(Facet, Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct A2aServer {
    pub id: String,
    pub name: String,
    pub description: String,
    pub input_modes: Vec<String>,
    pub output_modes: Vec<String>,
    pub streaming: bool,
    pub push_notifications: bool,
    pub enabled: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct A2aServerList {
    pub a2a_servers: Vec<A2aServer>,
}

#[derive(Serialize)]
pub struct CreateA2aServerRequest<'a> {
    pub base_url: &'a str,
}

#[derive(Serialize)]
struct A2aServerEnabledRequest {
    enabled: bool,
}

/// # Errors
/// 服务地址无效时返回固定错误信息。
pub fn get_a2a_servers<Effect, Event>(
    base_url: &str,
) -> Result<RequestBuilder<Effect, Event, A2aServerList>, String>
where
    Effect: From<Request<HttpRequest>> + Send + 'static,
    Event: Send + 'static,
{
    Ok(Http::get(endpoint(base_url, PATH)?).expect_json::<A2aServerList>())
}

/// # Errors
/// 服务地址无效或 JSON 编码失败时返回固定错误信息。
pub fn create_a2a_server<Effect, Event>(
    base_url: &str,
    remote_url: &str,
) -> Result<RequestBuilder<Effect, Event, ()>, String>
where
    Effect: From<Request<HttpRequest>> + Send + 'static,
    Event: Send + 'static,
{
    Http::post(endpoint(base_url, PATH)?)
        .body_json(&CreateA2aServerRequest {
            base_url: remote_url,
        })
        .map(|request| request.expect_json::<()>())
        .map_err(|_| "配置请求编码失败，请重试。".to_string())
}

/// # Errors
/// 服务地址、资源标识无效或 JSON 编码失败时返回固定错误信息。
pub fn set_a2a_server_enabled<Effect, Event>(
    base_url: &str,
    id: &str,
    enabled: bool,
) -> Result<RequestBuilder<Effect, Event, ()>, String>
where
    Effect: From<Request<HttpRequest>> + Send + 'static,
    Event: Send + 'static,
{
    Http::post(server_endpoint(base_url, id, "enabled")?)
        .body_json(&A2aServerEnabledRequest { enabled })
        .map(|request| request.expect_json::<()>())
        .map_err(|_| "配置请求编码失败，请重试。".to_string())
}

/// # Errors
/// 服务地址或资源标识无效时返回固定错误信息。
pub fn delete_a2a_server<Effect, Event>(
    base_url: &str,
    id: &str,
) -> Result<RequestBuilder<Effect, Event, ()>, String>
where
    Effect: From<Request<HttpRequest>> + Send + 'static,
    Event: Send + 'static,
{
    Ok(Http::post(server_endpoint(base_url, id, "delete")?).expect_json::<()>())
}

fn server_endpoint(base_url: &str, id: &str, action: &str) -> Result<Url, String> {
    if id.trim().is_empty() || matches!(id, "." | "..") {
        return Err("远程 Agent 标识无效，请刷新列表后重试。".to_string());
    }
    let mut url = endpoint(base_url, PATH)?;
    url.path_segments_mut()
        .map_err(|()| "接口地址无效，请检查 API 路径。".to_string())?
        .push(id)
        .push(action);
    Ok(url)
}

#[cfg(test)]
mod tests {
    use super::server_endpoint;

    #[test]
    fn encodes_ids_as_one_path_segment() {
        assert_eq!(
            server_endpoint(
                "https://api.example.test",
                "agent/../x?key=1#片段%2f",
                "delete"
            )
            .unwrap()
            .as_str(),
            "https://api.example.test/api/app_configs/a2a-servers/agent%2F..%2Fx%3Fkey=1%23%E7%89%87%E6%AE%B5%252f/delete"
        );
        for id in ["", " ", ".", ".."] {
            assert!(server_endpoint("https://api.example.test", id, "delete").is_err());
        }
    }
}
