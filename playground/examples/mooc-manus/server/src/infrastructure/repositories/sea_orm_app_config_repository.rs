use anyhow::Result;
use async_trait::async_trait;
use sea_orm::{DatabaseConnection, TransactionTrait};

use crate::{
    domain::{models::AppConfig, repositories::AppConfigRepository},
    models::{
        a2a_servers::A2aServers, agent_configs::AgentConfigs, llm_configs::LlmConfigs,
        mcp_servers::McpServers,
    },
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
        let mcp_config = McpServers::load_mcp_config(&self.db).await?;
        let a2a_config = A2aServers::load_a2a_config(&self.db).await?;

        if llm_config.is_none()
            && agent_config.is_none()
            && mcp_config.is_none()
            && a2a_config.is_none()
        {
            return Ok(None);
        }

        let mut config = AppConfig::default();
        if let Some(llm_config) = llm_config {
            config.llm_config = llm_config;
        }
        if let Some(agent_config) = agent_config {
            config.agent_config = agent_config;
        }
        if let Some(mcp_config) = mcp_config {
            config.mcp_config = mcp_config;
        }
        if let Some(a2a_config) = a2a_config {
            config.a2a_config = a2a_config;
        }

        Ok(Some(config))
    }

    async fn save(&self, config: AppConfig) -> Result<()> {
        self.db
            .transaction::<_, (), anyhow::Error>(|txn| {
                Box::pin(async move {
                    LlmConfigs::save_llm_config(txn, config.llm_config).await?;
                    AgentConfigs::save_agent_config(txn, config.agent_config).await?;
                    McpServers::save_mcp_config(txn, config.mcp_config).await?;
                    A2aServers::save_a2a_config(txn, config.a2a_config).await?;
                    Ok(())
                })
            })
            .await?;

        Ok(())
    }
}
