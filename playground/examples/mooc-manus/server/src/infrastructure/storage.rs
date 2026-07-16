use loco_rs::{
    app::AppContext,
    storage::{self as loco_storage, Storage},
    Error, Result,
};

use super::settings::{AppSettings, StorageSettings};

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

fn require_field(name: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(Error::Message(format!("{name} is required")));
    }

    Ok(())
}
