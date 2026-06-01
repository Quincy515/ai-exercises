use anyhow::Result;
use async_trait::async_trait;
use sea_orm::{DatabaseConnection, TransactionTrait};

use crate::{
    domain::{models::AppConfig, repositories::AppConfigRepository},
    models::{agent_configs::AgentConfigs, llm_configs::LlmConfigs},
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
        let llm_config = LlmConfigs::load_llm_config(&self.db).await?;
        let agent_config = AgentConfigs::load_agent_config(&self.db).await?;

        if llm_config.is_none() && agent_config.is_none() {
            return Ok(None);
        }

        let mut config = AppConfig::default();
        if let Some(llm_config) = llm_config {
            config.llm_config = llm_config;
        }
        if let Some(agent_config) = agent_config {
            config.agent_config = agent_config;
        }

        Ok(Some(config))
    }

    async fn save(&self, config: AppConfig) -> Result<()> {
        self.db
            .transaction::<_, (), anyhow::Error>(|txn| {
                Box::pin(async move {
                    LlmConfigs::save_llm_config(txn, config.llm_config).await?;
                    AgentConfigs::save_agent_config(txn, config.agent_config).await?;
                    Ok(())
                })
            })
            .await?;

        Ok(())
    }
}
