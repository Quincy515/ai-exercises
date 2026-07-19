//! A2A 服务配置的 Loco/SeaORM 模型门面。
//! Loco/SeaORM model facade for A2A server config persistence.

use std::collections::{HashMap, HashSet};

pub use super::_entities::a2a_servers::{ActiveModel, Column, Entity, Model};
use anyhow::{bail, Context, Result};
use sea_orm::entity::prelude::*;
use sea_orm::{
    ActiveModelTrait, ActiveValue::Set, ColumnTrait, Condition, EntityTrait, IntoActiveModel,
    QueryFilter, QueryOrder,
};
use uuid::Uuid;

use crate::domain::models::app_config::{A2aConfig, A2aServerConfig};

pub type A2aServers = Entity;

const STATUS_ENABLED: &str = "enabled";

/// 保存前钩子：更新已有记录时自动刷新 updated_at。
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

/// 已加载的 A2A 服务配置记录。
impl Model {
    /// 查找全部系统级 A2A 服务配置，包含软删除记录。
    pub async fn find_all_system_configs<C>(db: &C) -> Result<Vec<Self>>
    where
        C: ConnectionTrait,
    {
        let models = Entity::find()
            .filter(system_a2a_server_scope())
            .order_by_asc(Column::Id)
            .all(db)
            .await?;

        Ok(models)
    }

    /// 查找系统级 A2A 服务配置。
    ///
    /// 系统级配置用 `user_id IS NULL` 表示，并排除软删除记录。
    pub async fn find_system_configs<C>(db: &C) -> Result<Vec<Self>>
    where
        C: ConnectionTrait,
    {
        let models = Entity::find()
            .filter(system_a2a_server_condition())
            .order_by_asc(Column::Id)
            .all(db)
            .await?;

        Ok(models)
    }

    /// 将数据库行转换成领域层 A2A 服务配置。
    pub fn into_a2a_server_config(self) -> Result<A2aServerConfig> {
        Ok(A2aServerConfig {
            id: self.uuid.to_string(),
            base_url: self
                .base_url
                .context("a2a_servers.base_url must not be null")?,
            enabled: self.enabled.unwrap_or(true),
        })
    }
}

/// 准备写入的 A2A 服务配置记录。
impl ActiveModel {
    /// 从领域层 A2A 服务配置构造新的数据库记录。
    pub fn from_a2a_server_config(server_config: A2aServerConfig) -> Result<Self> {
        Ok(Self {
            uuid: Set(parse_a2a_id(&server_config.id)?),
            base_url: Set(Some(server_config.base_url)),
            enabled: Set(Some(server_config.enabled)),
            agent_card: Set(None),
            user_id: Set(None),
            status: Set(Some(STATUS_ENABLED.to_string())),
            is_deleted: Set(Some(false)),
            remark: Set(None),
            ..Default::default()
        })
    }

    /// 将领域层 A2A 服务配置应用到已有数据库记录。
    pub fn apply_a2a_server_config(&mut self, server_config: A2aServerConfig) -> Result<()> {
        self.uuid = Set(parse_a2a_id(&server_config.id)?);
        self.base_url = Set(Some(server_config.base_url));
        self.enabled = Set(Some(server_config.enabled));
        self.status = Set(Some(STATUS_ENABLED.to_string()));
        self.is_deleted = Set(Some(false));
        Ok(())
    }

    /// 软删除 A2A 服务配置。
    pub fn soft_delete(&mut self) {
        self.is_deleted = Set(Some(true));
    }
}

/// `a2a_servers` 表入口。
impl Entity {
    /// 从数据库加载 A2A 配置。
    pub async fn load_a2a_config<C>(db: &C) -> Result<Option<A2aConfig>>
    where
        C: ConnectionTrait,
    {
        let models = Model::find_system_configs(db).await?;
        if models.is_empty() {
            return Ok(None);
        }

        let a2a_servers = models
            .into_iter()
            .map(Model::into_a2a_server_config)
            .collect::<Result<Vec<_>>>()?;

        Ok(Some(A2aConfig { a2a_servers }))
    }

    /// 保存 A2A 配置。
    ///
    /// UUID 已存在时更新，没有记录时创建；配置中已移除的记录执行软删除。
    pub async fn save_a2a_config<C>(db: &C, a2a_config: A2aConfig) -> Result<()>
    where
        C: ConnectionTrait,
    {
        // 保存前完成 UUID 解析和去重，避免写入一半后才发现重复配置。
        let parsed_servers = parse_unique_a2a_servers(a2a_config.a2a_servers)?;
        // 保存时同时查询软删除记录，相同 UUID 可以恢复原记录并保留 agent_card。
        let mut existing_models = Model::find_all_system_configs(db)
            .await?
            .into_iter()
            .map(|model| (model.uuid, model))
            .collect::<HashMap<_, _>>();

        for (uuid, server_config) in parsed_servers {
            match existing_models.remove(&uuid) {
                Some(model) => {
                    let mut active_model = model.into_active_model();
                    active_model.apply_a2a_server_config(server_config)?;
                    active_model.update(db).await?;
                }
                None => {
                    ActiveModel::from_a2a_server_config(server_config)?
                        .insert(db)
                        .await?;
                }
            }
        }

        for model in existing_models.into_values() {
            if !model.is_deleted.unwrap_or(false) {
                let mut active_model = model.into_active_model();
                active_model.soft_delete();
                active_model.update(db).await?;
            }
        }

        Ok(())
    }
}

/// 构造全部系统级 A2A 服务配置查询条件。
fn system_a2a_server_scope() -> Condition {
    Condition::all().add(Column::UserId.is_null())
}

/// 构造系统级 A2A 服务配置查询条件。
fn system_a2a_server_condition() -> Condition {
    system_a2a_server_scope().add(
        Condition::any()
            .add(Column::IsDeleted.is_null())
            .add(Column::IsDeleted.eq(false)),
    )
}

fn parse_a2a_id(id: &str) -> Result<Uuid> {
    Uuid::parse_str(id).with_context(|| format!("invalid A2A server id: {id}"))
}

fn parse_unique_a2a_servers(servers: Vec<A2aServerConfig>) -> Result<Vec<(Uuid, A2aServerConfig)>> {
    let mut ids = HashSet::with_capacity(servers.len());
    let mut parsed_servers = Vec::with_capacity(servers.len());

    for server in servers {
        let uuid = parse_a2a_id(&server.id)?;
        if !ids.insert(uuid) {
            bail!("duplicate A2A server id: {}", server.id);
        }
        parsed_servers.push((uuid, server));
    }

    Ok(parsed_servers)
}
