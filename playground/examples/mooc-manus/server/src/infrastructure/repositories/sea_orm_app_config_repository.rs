use anyhow::Result;
use async_trait::async_trait;
use sea_orm::DatabaseConnection;

use crate::{
    domain::{models::AppConfig, repositories::AppConfigRepository},
    models::llm_configs::LlmConfigs,
};

pub struct SeaOrmAppConfigRepository {
    db: DatabaseConnection,
}

impl SeaOrmAppConfigRepository {
    pub fn new(db: DatabaseConnection) -> Self {
        Self { db }
    }
}

#[async_trait]
impl AppConfigRepository for SeaOrmAppConfigRepository {
    async fn load(&self) -> Result<Option<AppConfig>> {
        LlmConfigs::load_app_config(&self.db).await
    }

    async fn save(&self, config: AppConfig) -> Result<()> {
        LlmConfigs::save_app_config(&self.db, config).await
    }
}
