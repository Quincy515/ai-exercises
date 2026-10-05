//! 按后端业务模块组织客户端 API；各模块复用这里的公共请求工具。

pub mod configs;

use crux_http::Url;

/// Shell 提供服务地址，核心统一拼接从根路径开始的 API 地址。
pub(crate) fn endpoint(base_url: &str, path: &str) -> Result<Url, String> {
    let base =
        Url::parse(base_url).map_err(|_| "服务地址无效，请检查 API 地址配置。".to_string())?;

    if !matches!(base.scheme(), "http" | "https") || base.host_str().is_none() {
        return Err("服务地址必须使用 HTTP 或 HTTPS。".to_string());
    }

    base.join(path)
        .map_err(|_| "接口地址无效，请检查 API 路径。".to_string())
}

#[cfg(test)]
mod tests {
    use super::endpoint;

    #[test]
    fn joins_api_path_from_the_service_root() {
        for base in [
            "https://example.com",
            "https://example.com/",
            "https://example.com/settings?tab=agent",
        ] {
            assert_eq!(
                endpoint(base, "/api/app_configs/agent").unwrap().as_str(),
                "https://example.com/api/app_configs/agent"
            );
        }
    }

    #[test]
    fn rejects_invalid_and_non_http_service_addresses() {
        for base in [
            "",
            "/api",
            "invalid",
            "file:///tmp/api",
            "ftp://example.com",
        ] {
            assert!(endpoint(base, "/api/app_configs/agent").is_err(), "{base}");
        }
    }
}
