//! Agent 配置的 Loco/SeaORM 模型门面。
//! Loco/SeaORM model facade for Agent config persistence.
//!
//! `Model` 负责读取后的领域模型转换。
//! `Model` owns read-side conversion into the domain model.
//!
//! `ActiveModel` 负责创建和更新数据库字段。
//! `ActiveModel` owns create and update field mapping.
//!
//! `Entity` 负责系统级 Agent 配置的查询和保存流程。
//! `Entity` owns system-level Agent config queries and persistence.

pub use super::_entities::agent_configs::{ActiveModel, Column, Entity, Model};
use anyhow::{Context, Result};
use sea_orm::entity::prelude::*;
use sea_orm::{
    ActiveModelTrait, ActiveValue::Set, ColumnTrait, Condition, EntityTrait, IntoActiveModel,
    QueryFilter, QueryOrder,
};
use uuid::Uuid;

use crate::domain::models::AgentConfig;

pub type AgentConfigs = Entity;

const STATUS_ENABLED: &str = "enabled";

/// 保存前钩子：更新已有记录时自动刷新 updated_at。
/// Save hook: refresh updated_at automatically when updating an existing row.
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

/// 已加载的 Agent 配置记录。
/// Loaded Agent config row.
impl Model {
    /// 查找系统级 Agent 配置。
    /// Find the system-level Agent config row.
    ///
    /// 系统级配置用 `user_id IS NULL` 表示，并排除软删除记录。
    /// System-level config means `user_id IS NULL`, excluding soft-deleted rows.
    pub async fn find_system_config<C>(db: &C) -> Result<Option<Self>>
    where
        C: ConnectionTrait,
    {
        let model = Entity::find()
            .filter(system_agent_config_condition())
            .order_by_asc(Column::Id)
            .one(db)
            .await?;

        Ok(model)
    }

    /// 将数据库行转换成领域层 Agent 配置。
    /// Convert a database row into the domain Agent config.
    pub fn into_agent_config(self) -> Result<AgentConfig> {
        let defaults = AgentConfig::default();

        Ok(AgentConfig {
            max_iterations: db_value_to_domain(
                self.max_iterations,
                defaults.max_iterations,
                "max_iterations",
            )?,
            max_retries: db_value_to_domain(self.max_retries, defaults.max_retries, "max_retries")?,
            max_search_results: db_value_to_domain(
                self.max_search_results,
                defaults.max_search_results,
                "max_search_results",
            )?,
        })
    }
}

/// 准备写入的 Agent 配置记录。
/// Pending Agent config row for insert or update.
impl ActiveModel {
    /// 从领域层 Agent 配置构造新的数据库记录。
    /// Build a new database row from the domain Agent config.
    pub fn from_agent_config(agent_config: AgentConfig) -> Result<Self> {
        Ok(Self {
            max_iterations: Set(Some(domain_value_to_db(
                agent_config.max_iterations,
                "max_iterations",
            )?)),
            max_retries: Set(Some(domain_value_to_db(
                agent_config.max_retries,
                "max_retries",
            )?)),
            max_search_results: Set(Some(domain_value_to_db(
                agent_config.max_search_results,
                "max_search_results",
            )?)),
            uuid: Set(Uuid::new_v4()),
            user_id: Set(None),
            status: Set(Some(STATUS_ENABLED.to_string())),
            is_deleted: Set(Some(false)),
            remark: Set(None),
            ..Default::default()
        })
    }

    /// 将领域层 Agent 配置应用到已有数据库记录。
    /// Apply the domain Agent config to an existing database row.
    pub fn apply_agent_config(&mut self, agent_config: AgentConfig) -> Result<()> {
        self.max_iterations = Set(Some(domain_value_to_db(
            agent_config.max_iterations,
            "max_iterations",
        )?));
        self.max_retries = Set(Some(domain_value_to_db(
            agent_config.max_retries,
            "max_retries",
        )?));
        self.max_search_results = Set(Some(domain_value_to_db(
            agent_config.max_search_results,
            "max_search_results",
        )?));
        Ok(())
    }
}

/// `agent_configs` 表入口。
/// Entrypoint for the `agent_configs` table.
impl Entity {
    /// 从数据库加载 Agent 配置。
    /// Load Agent config from the database.
    pub async fn load_agent_config<C>(db: &C) -> Result<Option<AgentConfig>>
    where
        C: ConnectionTrait,
    {
        Model::find_system_config(db)
            .await?
            .map(Model::into_agent_config)
            .transpose()
    }

    /// 保存 Agent 配置。
    /// Save Agent config.
    ///
    /// 已存在系统级配置时更新，没有记录时创建。
    /// Update the system-level config when present, otherwise insert one.
    pub async fn save_agent_config<C>(db: &C, agent_config: AgentConfig) -> Result<()>
    where
        C: ConnectionTrait,
    {
        match Model::find_system_config(db).await? {
            Some(model) => {
                let mut active_model = model.into_active_model();
                active_model.apply_agent_config(agent_config)?;
                active_model.update(db).await?;
            }
            None => {
                ActiveModel::from_agent_config(agent_config)?
                    .insert(db)
                    .await?;
            }
        }

        Ok(())
    }
}

/// 构造系统级 Agent 配置查询条件。
/// Build the query condition for the system-level Agent config.
fn system_agent_config_condition() -> Condition {
    Condition::all().add(Column::UserId.is_null()).add(
        Condition::any()
            .add(Column::IsDeleted.is_null())
            .add(Column::IsDeleted.eq(false)),
    )
}

/// 将数据库的小整数转成领域层 `usize`。
/// Convert a database small integer into the domain `usize`.
fn db_value_to_domain(value: Option<i16>, default: usize, field: &str) -> Result<usize> {
    value.map_or(Ok(default), |value| {
        usize::try_from(value).with_context(|| {
            format!("agent_configs.{field} must be greater than or equal to 0: {value}")
        })
    })
}

/// 将领域层 `usize` 转成数据库的小整数。
/// Convert a domain `usize` into the database small integer.
fn domain_value_to_db(value: usize, field: &str) -> Result<i16> {
    i16::try_from(value)
        .with_context(|| format!("agent_config.{field} is larger than database i16 range: {value}"))
}
