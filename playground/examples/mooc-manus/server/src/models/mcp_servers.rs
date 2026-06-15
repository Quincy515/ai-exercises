//! MCP 服务配置的 Loco/SeaORM 模型门面。
//! Loco/SeaORM model facade for MCP server config persistence.

pub use super::_entities::mcp_servers::{ActiveModel, Column, Entity, Model};
use anyhow::{bail, Result};
use sea_orm::entity::prelude::*;
use sea_orm::{
    ActiveModelTrait, ActiveValue::Set, ColumnTrait, Condition, EntityTrait, IntoActiveModel,
    QueryFilter, QueryOrder,
};
use serde_json::{Map, Value};
use uuid::Uuid;

use crate::domain::models::{McpConfig, McpServerConfig, McpTransport};

pub type McpServers = Entity;

const STATUS_ENABLED: &str = "enabled";

#[async_trait::async_trait]
impl ActiveModelBehavior for ActiveModel {
    async fn before_save<C>(self, _db: &C, insert: bool) -> std::result::Result<Self, DbErr>
    where
        C: ConnectionTrait,
    {
        if !insert && self.updated_at.is_unchanged() {
            let mut this = self;
            this.updated_at = sea_orm::ActiveValue::Set(chrono::Utc::now().into());
            Ok(this)
        } else {
            Ok(self)
        }
    }
}

/// 已加载的 MCP 服务配置记录。
/// Loaded MCP server config row.
impl Model {
    /// 查找系统级 MCP 服务配置。
    /// Find system-level MCP server config rows.
    pub async fn find_system_configs<C>(db: &C) -> Result<Vec<Self>>
    where
        C: ConnectionTrait,
    {
        let models = Entity::find()
            .filter(system_mcp_server_condition())
            .order_by_asc(Column::Id)
            .all(db)
            .await?;

        Ok(models)
    }

    /// 将数据库行转换成领域层 MCP 服务配置。
    /// Convert a database row into the domain MCP server config.
    pub fn into_named_mcp_server_config(self) -> Result<(String, McpServerConfig)> {
        Ok((
            self.name,
            McpServerConfig {
                transport: db_transport_to_domain(&self.transport)?,
                enabled: self.enabled,
                description: self.description,
                env: db_json_to_object(self.env, "env")?,
                command: self.command,
                args: self.args,
                url: self.url,
                headers: db_json_to_object(self.headers, "headers")?,
            },
        ))
    }
}

/// 准备写入的 MCP 服务配置记录。
/// Pending MCP server config row for insert/update.
impl ActiveModel {
    /// 从领域层 MCP 服务配置构造新的数据库记录。
    /// Build a new database row from the domain MCP server config.
    pub fn from_named_mcp_server_config(name: String, server_config: McpServerConfig) -> Self {
        Self {
            uuid: Set(Uuid::new_v4()),
            user_id: Set(None),
            status: Set(Some(STATUS_ENABLED.to_string())),
            is_deleted: Set(Some(false)),
            remark: Set(None),
            name: Set(name),
            transport: Set(domain_transport_to_db(server_config.transport).to_string()),
            enabled: Set(server_config.enabled),
            description: Set(server_config.description),
            env: Set(server_config.env.map(Value::Object)),
            command: Set(server_config.command),
            args: Set(server_config.args),
            url: Set(server_config.url),
            headers: Set(server_config.headers.map(Value::Object)),
            ..Default::default()
        }
    }

    /// 将领域层 MCP 服务配置应用到已有数据库记录。
    /// Apply the domain MCP server config to an existing database row.
    pub fn apply_named_mcp_server_config(&mut self, name: String, server_config: McpServerConfig) {
        self.name = Set(name);
        self.status = Set(Some(STATUS_ENABLED.to_string()));
        self.is_deleted = Set(Some(false));
        self.transport = Set(domain_transport_to_db(server_config.transport).to_string());
        self.enabled = Set(server_config.enabled);
        self.description = Set(server_config.description);
        self.env = Set(server_config.env.map(Value::Object));
        self.command = Set(server_config.command);
        self.args = Set(server_config.args);
        self.url = Set(server_config.url);
        self.headers = Set(server_config.headers.map(Value::Object));
    }

    /// 软删除 MCP 服务配置。
    /// Soft-delete an MCP server config row.
    pub fn soft_delete(&mut self) {
        self.is_deleted = Set(Some(true));
    }
}

/// `mcp_servers` 表入口。
/// Entrypoint for the `mcp_servers` table.
impl Entity {
    /// 从数据库加载 MCP 配置。
    /// Load MCP config from the database.
    pub async fn load_mcp_config<C>(db: &C) -> Result<Option<McpConfig>>
    where
        C: ConnectionTrait,
    {
        let models = Model::find_system_configs(db).await?;
        if models.is_empty() {
            return Ok(None);
        }

        let mut mcp_config = McpConfig::default();
        for model in models {
            let (name, server_config) = model.into_named_mcp_server_config()?;
            mcp_config.mcp_servers.insert(name, server_config);
        }

        Ok(Some(mcp_config))
    }

    /// 保存 MCP 配置。
    /// Save MCP config.
    pub async fn save_mcp_config<C>(db: &C, mcp_config: McpConfig) -> Result<()>
    where
        C: ConnectionTrait,
    {
        let mut existing_models = Model::find_system_configs(db)
            .await?
            .into_iter()
            .map(|model| (model.name.clone(), model))
            .collect::<std::collections::HashMap<_, _>>();

        for (name, server_config) in mcp_config.mcp_servers {
            match existing_models.remove(&name) {
                Some(model) => {
                    let mut active_model = model.into_active_model();
                    active_model.apply_named_mcp_server_config(name, server_config);
                    active_model.update(db).await?;
                }
                None => {
                    ActiveModel::from_named_mcp_server_config(name, server_config)
                        .insert(db)
                        .await?;
                }
            }
        }

        for model in existing_models.into_values() {
            let mut active_model = model.into_active_model();
            active_model.soft_delete();
            active_model.update(db).await?;
        }

        Ok(())
    }
}

/// 构造系统级 MCP 服务配置查询条件。
/// Build the query condition for system-level MCP server configs.
fn system_mcp_server_condition() -> Condition {
    Condition::all().add(Column::UserId.is_null()).add(
        Condition::any()
            .add(Column::IsDeleted.is_null())
            .add(Column::IsDeleted.eq(false)),
    )
}

fn db_transport_to_domain(transport: &str) -> Result<McpTransport> {
    match transport {
        "stdio" => Ok(McpTransport::Stdio),
        "streamable_http" => Ok(McpTransport::StreamableHttp),
        value => bail!("unsupported mcp_servers.transport: {value}"),
    }
}

fn domain_transport_to_db(transport: McpTransport) -> &'static str {
    match transport {
        McpTransport::Stdio => "stdio",
        McpTransport::StreamableHttp => "streamable_http",
    }
}

fn db_json_to_object(value: Option<Value>, field: &str) -> Result<Option<Map<String, Value>>> {
    value
        .map(|value| match value {
            Value::Object(map) => Ok(map),
            _ => bail!("mcp_servers.{field} must be a JSON object"),
        })
        .transpose()
}
