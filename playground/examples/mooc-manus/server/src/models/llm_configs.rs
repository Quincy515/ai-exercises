//! LLM 配置的 Loco/SeaORM 模型门面。
//! Loco/SeaORM model facade for LLM config persistence.
//!
//! `Model` 表示一行已经从数据库读出的数据，适合放读取后的转换方法。
//! `Model` represents one loaded database row and owns read-side conversion helpers.
//!
//! `ActiveModel` 表示准备写入数据库的数据，适合放创建、更新字段的方法。
//! `ActiveModel` represents pending database writes and owns create/update helpers.
//!
//! `Entity` 表示 `llm_configs` 表入口，适合放表级查询和保存流程。
//! `Entity` represents the `llm_configs` table entrypoint and owns table-level operations.

pub use super::_entities::llm_configs::{ActiveModel, Column, Entity, Model};
use anyhow::{Context, Result};
use sea_orm::entity::prelude::*;
use sea_orm::{
    ActiveModelTrait, ActiveValue::Set, ColumnTrait, Condition, EntityTrait, IntoActiveModel,
    QueryFilter, QueryOrder,
};
use uuid::Uuid;

use crate::domain::models::LlmConfig;

pub type LlmConfigs = Entity;

const STATUS_ENABLED: &str = "enabled";

// 保存前钩子：更新已有记录时自动刷新 updated_at。
// Save hook: refresh updated_at automatically when updating an existing row.
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

/// 已加载的 LLM 配置记录。
/// Loaded LLM config row.
impl Model {
    /// 查找系统级 LLM 配置。
    /// Find the system-level LLM config row.
    ///
    /// 系统级配置当前用 `user_id IS NULL` 表示，并排除软删除记录。
    /// System-level config currently means `user_id IS NULL`, excluding soft-deleted rows.
    pub async fn find_system_config<C>(db: &C) -> Result<Option<Self>>
    where
        C: ConnectionTrait,
    {
        let model = Entity::find()
            .filter(system_llm_config_condition())
            .order_by_asc(Column::Id)
            .one(db)
            .await?;

        Ok(model)
    }

    /// 将数据库行转换成领域层 LLM 配置。
    /// Convert a database row into the domain LLM config.
    ///
    /// 这里集中处理数据库字段类型和领域字段类型的差异。
    /// This centralizes conversion between database field types and domain field types.
    pub fn into_llm_config(self) -> Result<LlmConfig> {
        Ok(LlmConfig {
            base_url: self.base_url,
            api_key: self.api_key,
            model_name: self.model_name,
            temperature: self.temperature,
            max_tokens: db_max_tokens_to_domain(self.max_tokens)?,
        })
    }
}

/// 准备写入的 LLM 配置记录。
/// Pending LLM config row for insert/update.
impl ActiveModel {
    /// 从领域层 LLM 配置构造一条新的数据库记录。
    /// Build a new database row from the domain LLM config.
    ///
    /// 新记录会补齐持久化层字段：`uuid`、`status`、`is_deleted` 等。
    /// New rows fill persistence fields such as `uuid`, `status`, and `is_deleted`.
    pub fn from_llm_config(llm_config: LlmConfig) -> Result<Self> {
        Ok(Self {
            base_url: Set(llm_config.base_url),
            api_key: Set(llm_config.api_key),
            model_name: Set(llm_config.model_name),
            temperature: Set(llm_config.temperature),
            max_tokens: Set(domain_max_tokens_to_db(llm_config.max_tokens)?),
            uuid: Set(Uuid::new_v4()),
            user_id: Set(None),
            status: Set(Some(STATUS_ENABLED.to_string())),
            is_deleted: Set(Some(false)),
            remark: Set(None),
            ..Default::default()
        })
    }

    /// 将领域层 LLM 配置应用到已有数据库记录。
    /// Apply the domain LLM config to an existing database row.
    ///
    /// 这里只更新业务配置字段，保留 `uuid`、`user_id`、状态等持久化元数据。
    /// This updates config fields while preserving persistence metadata such as `uuid`, `user_id`, and status.
    pub fn apply_llm_config(&mut self, llm_config: LlmConfig) -> Result<()> {
        self.base_url = Set(llm_config.base_url);
        self.api_key = Set(llm_config.api_key);
        self.model_name = Set(llm_config.model_name);
        self.temperature = Set(llm_config.temperature);
        self.max_tokens = Set(domain_max_tokens_to_db(llm_config.max_tokens)?);
        Ok(())
    }
}

/// `llm_configs` 表入口。
/// Entrypoint for the `llm_configs` table.
impl Entity {
    /// 从数据库加载 LLM 配置。
    /// Load LLM config from the database.
    pub async fn load_llm_config<C>(db: &C) -> Result<Option<LlmConfig>>
    where
        C: ConnectionTrait,
    {
        Model::find_system_config(db)
            .await?
            .map(Model::into_llm_config)
            .transpose()
    }

    /// 保存 LLM 配置。
    /// Save LLM config.
    ///
    /// 已存在系统级 LLM 配置时更新；没有记录时创建一条新的系统级配置。
    /// Updates the existing system-level LLM config, or inserts a new one when missing.
    pub async fn save_llm_config<C>(db: &C, llm_config: LlmConfig) -> Result<()>
    where
        C: ConnectionTrait,
    {
        match Model::find_system_config(db).await? {
            Some(model) => {
                let mut active_model = model.into_active_model();
                active_model.apply_llm_config(llm_config)?;
                active_model.update(db).await?;
            }
            None => {
                ActiveModel::from_llm_config(llm_config)?.insert(db).await?;
            }
        }

        Ok(())
    }
}

/// 构造系统级 LLM 配置查询条件。
/// Build the query condition for the system-level LLM config.
fn system_llm_config_condition() -> Condition {
    Condition::all().add(Column::UserId.is_null()).add(
        Condition::any()
            .add(Column::IsDeleted.is_null())
            .add(Column::IsDeleted.eq(false)),
    )
}

/// 将数据库里的 `max_tokens` 转成领域层类型。
/// Convert database `max_tokens` into the domain type.
fn db_max_tokens_to_domain(max_tokens: Option<i64>) -> Result<Option<usize>> {
    max_tokens
        .map(|value| {
            usize::try_from(value).with_context(|| {
                format!("llm_configs.max_tokens must be greater than or equal to 0: {value}")
            })
        })
        .transpose()
}

/// 将领域层 `max_tokens` 转成数据库类型。
/// Convert domain `max_tokens` into the database type.
fn domain_max_tokens_to_db(max_tokens: Option<usize>) -> Result<Option<i64>> {
    max_tokens
        .map(|value| {
            i64::try_from(value).with_context(|| {
                format!("llm_config.max_tokens is larger than database i64 range: {value}")
            })
        })
        .transpose()
}
