//! JSON 草稿校验；原始字段与解析错误均保持在内部。

use std::collections::BTreeMap;

use crux_http::Url;
use serde_json::{Map, Value};

use crate::api::configs::mcp::{McpConfigRequest, McpServerAck, valid_server_name};

type ParsedConfig = (McpConfigRequest, BTreeMap<String, McpServerAck>);

pub(super) fn parse_config(value: &str) -> Result<ParsedConfig, String> {
    let mut request: McpConfigRequest = serde_json::from_str(value)
        .map_err(|_| "请输入有效 JSON，根节点只包含 mcpServers 对象。".to_string())?;
    if request.mcp_servers.is_empty() {
        return Err("mcpServers 至少需要一个服务器配置。".to_string());
    }
    let mut expected = BTreeMap::new();
    for (name, config) in &mut request.mcp_servers {
        if !valid_server_name(name) {
            return Err("MCP 服务名称不能为空、包含控制字符或为 . / ..。".to_string());
        }
        let config = config
            .as_object_mut()
            .ok_or_else(|| "每个 MCP 服务器配置必须是 JSON 对象。".to_string())?;
        let summary = validate_server(config)?;
        expected.insert(name.clone(), summary);
    }
    Ok((request, expected))
}

fn validate_server(config: &mut Map<String, Value>) -> Result<McpServerAck, String> {
    if config.contains_key("disabled") {
        return Err("请使用 enabled 布尔字段控制启停，当前接口不支持 disabled。".to_string());
    }
    if config.keys().any(|key| {
        !matches!(
            key.as_str(),
            "transport"
                | "enabled"
                | "description"
                | "env"
                | "command"
                | "args"
                | "url"
                | "headers"
        )
    }) {
        return Err("MCP 配置包含未知字段，请核对 transport、enabled、command、args、env、url、headers、description 的拼写。".to_string());
    }
    let transport = match config.get("transport") {
        None => "streamable_http",
        Some(Value::String(value)) if matches!(value.as_str(), "stdio" | "streamable_http") => {
            value
        }
        _ => {
            return Err(
                "transport 仅支持 stdio 或 streamable_http，省略时使用 streamable_http。"
                    .to_string(),
            );
        }
    }
    .to_string();
    let enabled = match config.get("enabled") {
        None => true,
        Some(Value::Bool(value)) => *value,
        _ => return Err("enabled 必须是布尔值 true 或 false。".to_string()),
    };
    for key in ["description", "command", "url"] {
        if config
            .get(key)
            .is_some_and(|value| !value.is_null() && !value.is_string())
        {
            return Err("description、command 和 url 必须是字符串或 null。".to_string());
        }
    }
    if config.get("args").is_some_and(|value| {
        !value.is_null()
            && !value
                .as_array()
                .is_some_and(|items| items.iter().all(Value::is_string))
    }) {
        return Err("args 必须是字符串数组或 null。".to_string());
    }
    validate_env(config.get("env"))?;
    validate_headers(config.get("headers"))?;
    if transport == "stdio" {
        if config
            .get("command")
            .and_then(Value::as_str)
            .is_none_or(|value| value.trim().is_empty() || value.contains('\0'))
        {
            return Err("stdio 配置需要非空 command，请填写后端可执行的命令。".to_string());
        }
    } else {
        let valid = config
            .get("url")
            .and_then(Value::as_str)
            .is_some_and(|value| {
                !value.chars().any(char::is_control)
                    && Url::parse(value).ok().is_some_and(|url| {
                        matches!(url.scheme(), "http" | "https")
                            && url.host_str().is_some()
                            && url.username().is_empty()
                            && url.password().is_none()
                            && url.fragment().is_none()
                    })
            });
        if !valid {
            return Err(
                "streamable_http 配置需要 HTTP 或 HTTPS url，且不能包含用户名、密码或片段。"
                    .to_string(),
            );
        }
    }
    config.insert("transport".to_string(), Value::String(transport.clone()));
    config.insert("enabled".to_string(), Value::Bool(enabled));
    Ok(McpServerAck { transport, enabled })
}

fn validate_env(value: Option<&Value>) -> Result<(), String> {
    if value.is_none_or(Value::is_null) {
        return Ok(());
    }
    let valid = value.and_then(Value::as_object).is_some_and(|values| {
        values.iter().all(|(key, value)| {
            !key.is_empty()
                && !key.contains(['=', '\0'])
                && value.as_str().is_some_and(|text| !text.contains('\0'))
        })
    });
    if valid {
        Ok(())
    } else {
        Err("env 必须是字符串键值对象或 null；名称须非空，名称和值须适合环境变量。".to_string())
    }
}

fn validate_headers(value: Option<&Value>) -> Result<(), String> {
    if value.is_none_or(Value::is_null) {
        return Ok(());
    }
    let valid = value.and_then(Value::as_object).is_some_and(|values| {
        values.iter().all(|(key, value)| {
            !key.is_empty()
                && key
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&byte))
                && value.as_str().is_some_and(|text| {
                    text.bytes()
                        .all(|byte| byte == b'\t' || byte >= 0x20 && byte != 0x7f)
                })
        })
    });
    if valid {
        Ok(())
    } else {
        Err("headers 必须是字符串键值对象或 null；请求头名称和值须符合 HTTP 格式。".to_string())
    }
}
