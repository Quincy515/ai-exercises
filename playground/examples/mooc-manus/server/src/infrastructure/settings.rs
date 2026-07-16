use std::path::PathBuf;

use loco_rs::{config::Config, Result};
use serde::{Deserialize, Deserializer, Serialize};

/// Loco `settings` 下的应用自定义配置。
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct AppSettings {
    #[serde(default)]
    pub storage: StorageSettings,
    #[serde(default)]
    pub sandbox: SandboxSettings,
}

impl AppSettings {
    /// 从 Loco 的动态 `settings` 配置中读取强类型应用配置。
    pub fn from_config(config: &Config) -> Result<Self> {
        config
            .settings
            .as_ref()
            .map_or_else(|| Ok(Self::default()), Self::from_json)
    }

    pub fn from_json(settings: &serde_json::Value) -> Result<Self> {
        Ok(serde_json::from_value(settings.clone())?)
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(tag = "kind")]
pub enum StorageSettings {
    #[default]
    #[serde(rename = "null", alias = "Null")]
    Null,
    #[serde(rename = "memory", alias = "Memory", alias = "mem")]
    Memory,
    #[serde(rename = "local", alias = "Local")]
    Local { path: PathBuf },
    #[serde(rename = "r2", alias = "R2", alias = "cloudflare_r2")]
    R2 {
        account_id: String,
        bucket: String,
        access_key_id: String,
        secret_access_key: String,
        #[serde(default = "default_r2_region")]
        region: String,
    },
}

/// Sandbox 配置。
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct SandboxSettings {
    /// 已有沙箱地址。存在时直接连接，留空时由 Docker 适配器创建容器。
    #[serde(default, deserialize_with = "empty_string_as_none")]
    pub address: Option<String>,
    /// 创建沙箱容器时使用的镜像。
    #[serde(default, deserialize_with = "empty_string_as_none")]
    pub image: Option<String>,
    /// 沙箱容器名称前缀。
    #[serde(default, deserialize_with = "empty_string_as_none")]
    pub name_prefix: Option<String>,
    /// 沙箱过期时间，单位为分钟。
    #[serde(default = "default_sandbox_ttl_minutes")]
    pub ttl_minutes: Option<u64>,
    /// 沙箱 API 与其他服务共同使用的 Docker 网络。
    #[serde(default, deserialize_with = "empty_string_as_none")]
    pub network: Option<String>,
    /// 传递给沙箱中 Chrome 的启动参数。
    #[serde(default, deserialize_with = "empty_string_as_none")]
    pub chrome_args: Option<String>,
    #[serde(default, deserialize_with = "empty_string_as_none")]
    pub https_proxy: Option<String>,
    #[serde(default, deserialize_with = "empty_string_as_none")]
    pub http_proxy: Option<String>,
    #[serde(default, deserialize_with = "empty_string_as_none")]
    pub no_proxy: Option<String>,
}

impl Default for SandboxSettings {
    fn default() -> Self {
        Self {
            address: None,
            image: None,
            name_prefix: None,
            ttl_minutes: default_sandbox_ttl_minutes(),
            network: None,
            chrome_args: None,
            https_proxy: None,
            http_proxy: None,
            no_proxy: None,
        }
    }
}

fn default_r2_region() -> String {
    "auto".to_string()
}

const fn default_sandbox_ttl_minutes() -> Option<u64> {
    Some(60)
}

fn empty_string_as_none<'de, D>(deserializer: D) -> std::result::Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<String>::deserialize(deserializer)?;
    Ok(value.filter(|value| !value.trim().is_empty()))
}

#[cfg(test)]
mod tests {
    use super::{AppSettings, SandboxSettings, StorageSettings};
    use loco_rs::{config::Config, environment::Environment};
    use serde_json::json;
    use std::path::PathBuf;

    #[test]
    fn parses_memory_storage_and_sandbox_settings() {
        let settings = AppSettings::from_json(&json!({
            "storage": {
                "kind": "memory"
            },
            "sandbox": {
                "address": "",
                "image": "mooc-manus-sandbox",
                "name_prefix": "mooc-manus-sandbox",
                "ttl_minutes": 90,
                "network": "mooc-manus-network",
                "chrome_args": "--headless=new",
                "https_proxy": " ",
                "http_proxy": null,
                "no_proxy": "localhost,127.0.0.1"
            }
        }))
        .unwrap();

        assert!(matches!(settings.storage, StorageSettings::Memory));
        assert_eq!(
            settings.sandbox,
            SandboxSettings {
                address: None,
                image: Some("mooc-manus-sandbox".to_string()),
                name_prefix: Some("mooc-manus-sandbox".to_string()),
                ttl_minutes: Some(90),
                network: Some("mooc-manus-network".to_string()),
                chrome_args: Some("--headless=new".to_string()),
                https_proxy: None,
                http_proxy: None,
                no_proxy: Some("localhost,127.0.0.1".to_string()),
            }
        );
    }

    #[test]
    fn applies_defaults_when_custom_settings_are_missing() {
        let settings = AppSettings::from_json(&json!({})).unwrap();

        assert!(matches!(settings.storage, StorageSettings::Null));
        assert_eq!(settings.sandbox, SandboxSettings::default());
        assert_eq!(settings.sandbox.ttl_minutes, Some(60));
    }

    #[test]
    fn loads_sandbox_settings_from_development_config() {
        let config_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("config");
        let config = Config::from_folder(&Environment::Development, &config_dir).unwrap();
        let settings = AppSettings::from_config(&config).unwrap();

        assert_eq!(
            settings.sandbox,
            SandboxSettings {
                address: optional_env("SANDBOX_ADDRESS", ""),
                image: optional_env("SANDBOX_IMAGE", "lenexus-sandbox"),
                name_prefix: optional_env("SANDBOX_NAME_PREFIX", "lenexus-sandbox"),
                ttl_minutes: ttl_minutes_from_env(),
                network: optional_env("SANDBOX_NETWORK", "lenexus-network"),
                chrome_args: optional_env("SANDBOX_CHROME_ARGS", ""),
                https_proxy: optional_env("SANDBOX_HTTPS_PROXY", ""),
                http_proxy: optional_env("SANDBOX_HTTP_PROXY", ""),
                no_proxy: optional_env("SANDBOX_NO_PROXY", ""),
            }
        );
    }

    #[test]
    fn parses_r2_storage_settings_with_default_region() {
        let settings = AppSettings::from_json(&json!({
            "storage": {
                "kind": "r2",
                "account_id": "account",
                "bucket": "bucket",
                "access_key_id": "key",
                "secret_access_key": "secret"
            }
        }))
        .unwrap();

        let StorageSettings::R2 {
            account_id,
            bucket,
            region,
            ..
        } = settings.storage
        else {
            panic!("expected r2 storage settings");
        };

        assert_eq!(account_id, "account");
        assert_eq!(bucket, "bucket");
        assert_eq!(region, "auto");
    }

    fn optional_env(name: &str, default: &str) -> Option<String> {
        let value = std::env::var(name).unwrap_or_else(|_| default.to_string());
        (!value.trim().is_empty()).then_some(value)
    }

    fn ttl_minutes_from_env() -> Option<u64> {
        match std::env::var("SANDBOX_TTL_MINUTES") {
            Ok(value) if value.trim().is_empty() => None,
            Ok(value) => Some(value.parse().unwrap()),
            Err(_) => Some(60),
        }
    }
}
