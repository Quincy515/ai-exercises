use std::path::PathBuf;

use loco_rs::{
    app::AppContext,
    config::Config,
    storage::{self as loco_storage, Storage},
    Error, Result,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct AppSettings {
    #[serde(default)]
    pub storage: StorageSettings,
}

impl AppSettings {
    pub fn from_config(config: &Config) -> Result<Self> {
        config.settings.as_ref().map_or_else(
            || Ok(Self::default()),
            |settings| Ok(serde_json::from_value(settings.clone())?),
        )
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

impl StorageSettings {
    pub fn build(&self) -> Result<Storage> {
        let driver = match self {
            Self::Null => loco_storage::drivers::null::new(),
            Self::Memory => loco_storage::drivers::mem::new(),
            Self::Local { path } => loco_storage::drivers::local::new_with_prefix(path)?,
            Self::R2 {
                account_id,
                bucket,
                access_key_id,
                secret_access_key,
                region,
            } => {
                require_field("settings.storage.account_id", account_id)?;
                require_field("settings.storage.bucket", bucket)?;
                require_field("settings.storage.access_key_id", access_key_id)?;
                require_field("settings.storage.secret_access_key", secret_access_key)?;

                let endpoint = format!("https://{account_id}.r2.cloudflarestorage.com");
                let credential = loco_storage::drivers::aws::Credential {
                    key_id: access_key_id.clone(),
                    secret_key: secret_access_key.clone(),
                    token: None,
                };

                loco_storage::drivers::aws::with_credentials_and_endpoint(
                    bucket, region, &endpoint, credential,
                )?
            }
        };

        Ok(Storage::single(driver))
    }
}

pub async fn configure_storage(ctx: AppContext) -> Result<AppContext> {
    let settings = AppSettings::from_config(&ctx.config)?;
    let storage = settings.storage.build()?;

    Ok(AppContext {
        storage: storage.into(),
        ..ctx
    })
}

fn default_r2_region() -> String {
    "auto".to_string()
}

fn require_field(name: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(Error::Message(format!("{name} is required")));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{AppSettings, StorageSettings};
    use serde_json::json;

    #[test]
    fn parses_memory_storage_settings() {
        let settings: AppSettings = serde_json::from_value(json!({
            "storage": {
                "kind": "memory"
            }
        }))
        .unwrap();

        assert!(matches!(settings.storage, StorageSettings::Memory));
    }

    #[test]
    fn parses_r2_storage_settings_with_default_region() {
        let settings: AppSettings = serde_json::from_value(json!({
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
}
