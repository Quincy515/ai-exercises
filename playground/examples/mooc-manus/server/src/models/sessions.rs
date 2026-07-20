//! 会话的 Loco/SeaORM 模型门面。

pub use super::_entities::sessions::{ActiveModel, Column, Entity, Model};
use anyhow::{bail, Context, Result};
use sea_orm::entity::prelude::*;
use sea_orm::ActiveValue::Set;
use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::domain::models::{Session, SessionStatus};

pub type Sessions = Entity;

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

/// 已加载的会话记录。
impl Model {
    /// 将会话 ORM 模型转换成领域模型。
    pub fn into_session(self) -> Result<Session> {
        let uuid = self.uuid.context("sessions.uuid must not be null")?;
        let unread_message_count =
            usize::try_from(self.unread_message_count.unwrap_or_default())
                .context("sessions.unread_message_count must be greater than or equal to 0")?;

        Ok(Session {
            id: uuid.to_string(),
            sandbox_id: self.sandbox_id,
            task_id: self.task_id,
            title: self.title.unwrap_or_default(),
            unread_message_count,
            latest_message: self.latest_message.unwrap_or_default(),
            latest_message_at: self
                .latest_message_at
                .map(|value| value.with_timezone(&chrono::Utc)),
            events: deserialize_json(self.events, Value::Array(Vec::new()), "sessions.events")?,
            files: deserialize_json(self.files, Value::Array(Vec::new()), "sessions.files")?,
            memories: deserialize_json(
                self.memories,
                Value::Object(Default::default()),
                "sessions.memories",
            )?,
            status: session_status_from_db(self.status.as_deref())?,
            updated_at: self.updated_at.with_timezone(&chrono::Utc),
            created_at: self.created_at.with_timezone(&chrono::Utc),
        })
    }
}

/// 准备写入的会话记录。
impl ActiveModel {
    /// 从会话领域模型构建 ORM 模型。
    pub fn from_session(session: Session) -> Result<Self> {
        let uuid = Uuid::parse_str(&session.id)
            .with_context(|| format!("invalid session id: {}", session.id))?;
        let unread_message_count =
            i32::try_from(session.unread_message_count).with_context(|| {
                format!(
                    "session.unread_message_count is larger than database i32 range: {}",
                    session.unread_message_count
                )
            })?;
        let events = serialize_json(&session.events, "session.events")?;
        let files = serialize_json(&session.files, "session.files")?;
        let memories = serialize_json(&session.memories, "session.memories")?;

        Ok(Self {
            uuid: Set(Some(uuid)),
            sandbox_id: Set(session.sandbox_id),
            task_id: Set(session.task_id),
            title: Set(Some(session.title)),
            unread_message_count: Set(Some(unread_message_count)),
            latest_message: Set(Some(session.latest_message)),
            latest_message_at: Set(session.latest_message_at.map(Into::into)),
            events: Set(Some(events)),
            files: Set(Some(files)),
            memories: Set(Some(memories)),
            status: Set(Some(session.status.as_str().to_string())),
            user_id: Set(None),
            is_deleted: Set(Some(false)),
            remark: Set(None),
            ..Default::default()
        })
    }

    /// 从传递的领域模型更新 ORM 数据。
    pub fn apply_session(&mut self, session: Session) -> Result<()> {
        let source = Self::from_session(session)?;

        self.uuid = source.uuid;
        self.sandbox_id = source.sandbox_id;
        self.task_id = source.task_id;
        self.title = source.title;
        self.unread_message_count = source.unread_message_count;
        self.latest_message = source.latest_message;
        self.latest_message_at = source.latest_message_at;
        self.events = source.events;
        self.files = source.files;
        self.memories = source.memories;
        self.status = source.status;

        Ok(())
    }
}

// implement your custom finders, selectors oriented logic here
impl Entity {}

fn deserialize_json<T>(value: Option<Json>, default: Value, field: &str) -> Result<T>
where
    T: DeserializeOwned,
{
    serde_json::from_value(value.unwrap_or(default))
        .with_context(|| format!("failed to deserialize {field}"))
}

fn serialize_json<T>(value: &T, field: &str) -> Result<Json>
where
    T: Serialize,
{
    serde_json::to_value(value).with_context(|| format!("failed to serialize {field}"))
}

fn session_status_from_db(status: Option<&str>) -> Result<SessionStatus> {
    match status {
        None | Some("") | Some("pending") => Ok(SessionStatus::Pending),
        Some("running") => Ok(SessionStatus::Running),
        Some("waiting") => Ok(SessionStatus::Waiting),
        Some("completed") => Ok(SessionStatus::Completed),
        Some(status) => bail!("invalid sessions.status: {status}"),
    }
}
