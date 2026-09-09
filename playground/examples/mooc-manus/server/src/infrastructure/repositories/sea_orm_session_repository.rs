use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use sea_orm::{
    sea_query::{Expr, NullOrdering, OnConflict},
    ActiveModelTrait,
    ActiveValue::Set,
    ColumnTrait, DatabaseConnection, EntityTrait, IntoActiveModel, Order, QueryFilter, QueryOrder,
    QuerySelect, TransactionTrait, UpdateMany,
};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::{
    domain::{
        models::{Event, File, Memory, Session, SessionStatus},
        repositories::SessionRepository,
    },
    models::sessions::{ActiveModel, Column, Entity as Sessions, Model},
};

/// 基于 PostgreSQL 数据库的会话仓库。
///
/// 只持有连接池句柄；每次普通写入由数据库提交，多语句操作显式使用事务。
/// JSONB 表达式和行锁依赖 PostgreSQL，不使用读取整份会话再覆盖的方式追加数据。
pub struct SeaOrmSessionRepository {
    db: DatabaseConnection,
}

impl SeaOrmSessionRepository {
    /// 构造函数，复用应用已经建立的数据库连接池。
    pub fn new(db: DatabaseConnection) -> Self {
        Self { db }
    }

    /// 为局部更新统一添加会话条件、更新时间和不存在检查。
    async fn update_session(&self, session_id: &str, update: UpdateMany<Sessions>) -> Result<()> {
        // 1.领域会话 id 对应 uuid 列；禁止误用数据库自增主键。
        let uuid = parse_session_id(session_id)?;

        // 2.update_many 不经过 ActiveModel 的保存钩子，需要显式刷新更新时间。
        let result = update
            .filter(Column::Uuid.eq(uuid))
            .col_expr(Column::UpdatedAt, Expr::current_timestamp().into())
            .exec(&self.db)
            .await?;

        // 3.即使列值没有变化，PostgreSQL 也会统计匹配行；零行表示会话不存在。
        if result.rows_affected == 0 {
            bail!("会话[{session_id}]不存在，请核实后重试");
        }
        Ok(())
    }
}

#[async_trait]
impl SessionRepository for SeaOrmSessionRepository {
    /// 根据传递的领域模型更新或者新增会话。
    async fn save(&self, session: Session) -> Result<()> {
        // 1.转换为 ORM 模型，同时检查 UUID、计数范围并序列化 JSONB。
        let record = ActiveModel::from_session(session)?;

        // 2.由 uuid 唯一索引判断新增或更新，避免先查询再插入时的并发竞争。
        // 仅覆盖领域业务字段，保留既有记录的主键、创建时间、用户、备注和删除标记。
        let on_conflict = OnConflict::column(Column::Uuid)
            .update_columns([
                Column::SandboxId,
                Column::TaskId,
                Column::Title,
                Column::UnreadMessageCount,
                Column::LatestMessage,
                Column::LatestMessageAt,
                Column::Events,
                Column::Files,
                Column::Memories,
                Column::Status,
            ])
            .value(Column::UpdatedAt, Expr::current_timestamp())
            .to_owned();

        // 3.一条 INSERT ... ON CONFLICT 完成保存，数据库负责原子提交。
        Sessions::insert(record)
            .on_conflict(on_conflict)
            .exec(&self.db)
            .await?;
        Ok(())
    }

    /// 获取所有会话列表，最近有消息的会话排在前面。
    async fn get_all(&self) -> Result<Vec<Session>> {
        // 1.按最新消息时间倒序；尚无消息的会话放在最后，再用创建时间和主键稳定排序。
        let records = Sessions::find()
            .order_by_with_nulls(Column::LatestMessageAt, Order::Desc, NullOrdering::Last)
            .order_by_desc(Column::CreatedAt)
            .order_by_desc(Column::Id)
            .all(&self.db)
            .await?;

        // 2.将每条 ORM 记录转换成领域对象，任何转换失败都向上传播。
        records.into_iter().map(Model::into_session).collect()
    }

    /// 根据领域会话 id 查询会话。
    async fn get_by_id(&self, session_id: &str) -> Result<Option<Session>> {
        // 1.按 uuid 查询零条或一条记录。
        let record = Sessions::find()
            .filter(Column::Uuid.eq(parse_session_id(session_id)?))
            .one(&self.db)
            .await?;

        // 2.存在则转换领域模型，不存在则返回 None；转换错误不能吞成 None。
        record.map(Model::into_session).transpose()
    }

    /// 根据会话 id 物理删除记录；重复删除无需报错。
    async fn delete_by_id(&self, session_id: &str) -> Result<()> {
        // 1.构建删除语句，条件使用领域 id 对应的 uuid。
        // 2.执行删除，不检查受影响行数，保持删除操作幂等。
        Sessions::delete_many()
            .filter(Column::Uuid.eq(parse_session_id(session_id)?))
            .exec(&self.db)
            .await?;
        Ok(())
    }

    /// 更新会话标题。
    async fn update_title(&self, session_id: &str, title: &str) -> Result<()> {
        // 1.只设置标题列，避免覆盖事件、文件和记忆。
        let update = Sessions::update_many().col_expr(Column::Title, Expr::value(title));
        // 2.统一执行更新并检查会话是否存在。
        self.update_session(session_id, update).await
    }

    /// 同时更新会话最新消息与消息发生时间。
    async fn update_latest_message(
        &self,
        session_id: &str,
        message: &str,
        timestamp: DateTime<Utc>,
    ) -> Result<()> {
        // 1.两个字段放在同一条更新语句中，避免消息与时间出现不一致。
        let update = Sessions::update_many()
            .col_expr(Column::LatestMessage, Expr::value(message))
            .col_expr(Column::LatestMessageAt, Expr::value(timestamp));
        // 2.执行更新并检查是否成功。
        self.update_session(session_id, update).await
    }

    /// 设置会话的未读消息数。
    async fn update_unread_message_count(&self, session_id: &str, count: usize) -> Result<()> {
        // 1.领域计数是 usize，但数据库列是 i32，写入前必须检查上界。
        let count = i32::try_from(count).context("未读消息数超过数据库 i32 范围")?;
        let update =
            Sessions::update_many().col_expr(Column::UnreadMessageCount, Expr::value(count));
        // 2.执行更新并检查是否成功。
        self.update_session(session_id, update).await
    }

    /// 原子增加一条未读消息。
    async fn increment_unread_message_count(&self, session_id: &str) -> Result<()> {
        // 1.NULL 按零处理，在数据库当前值上加一，避免并发读改写丢失计数。
        // i32 已达到上界时由 PostgreSQL 返回溢出错误，整条语句不会写入。
        let update = Sessions::update_many().col_expr(
            Column::UnreadMessageCount,
            Expr::cust("COALESCE(unread_message_count, 0) + 1"),
        );
        // 2.执行更新并检查是否成功。
        self.update_session(session_id, update).await
    }

    /// 原子减少一条未读消息，最小值为零。
    async fn decrement_unread_message_count(&self, session_id: &str) -> Result<()> {
        // 1.先将当前值夹到至少一，再减一，零值和历史负数均会恢复为零。
        // 这样也避免历史 i32::MIN 在执行减法时先溢出。
        let update = Sessions::update_many().col_expr(
            Column::UnreadMessageCount,
            Expr::cust("GREATEST(COALESCE(unread_message_count, 0), 1) - 1"),
        );
        // 2.执行更新并检查是否成功。
        self.update_session(session_id, update).await
    }

    /// 更新会话状态。
    async fn update_status(&self, session_id: &str, status: SessionStatus) -> Result<()> {
        // 1.领域枚举转换为数据库存储的小写字符串。
        let update = Sessions::update_many().col_expr(Column::Status, Expr::value(status.as_str()));
        // 2.执行更新并检查是否成功。
        self.update_session(session_id, update).await
    }

    /// 往会话中追加完整事件。
    async fn add_event(&self, session_id: &str, event: Event) -> Result<()> {
        // 1.序列化事件并包装成单元素数组，保留变体内的完整内容。
        let event_data = serde_json::to_value(vec![event]).context("序列化会话事件失败")?;
        // 2.JSONB 数组在数据库端拼接；事件内容使用绑定参数，不能拼进 SQL 字符串。
        let update = Sessions::update_many().col_expr(
            Column::Events,
            Expr::cust_with_values("COALESCE(events, '[]'::jsonb) || $1::jsonb", [event_data]),
        );
        // 3.原子更新并检查会话是否存在。
        self.update_session(session_id, update).await
    }

    /// 往会话中新增文件元数据，不负责上传文件内容。
    async fn add_file(&self, session_id: &str, file: File) -> Result<()> {
        // 1.将文件元数据序列化成单元素数组。
        let file_data = serde_json::to_value(vec![file]).context("序列化会话文件失败")?;
        // 2.原子追加 JSONB 数组，NULL 文件列表按空数组处理。
        let update = Sessions::update_many().col_expr(
            Column::Files,
            Expr::cust_with_values("COALESCE(files, '[]'::jsonb) || $1::jsonb", [file_data]),
        );
        // 3.执行更新并检查是否成功。
        self.update_session(session_id, update).await
    }

    /// 移除会话中的指定文件记录，不删除真实文件。
    async fn remove_file(&self, session_id: &str, file_id: &str) -> Result<()> {
        // 1.开启事务并锁住会话行，直到过滤结果写回才释放锁。
        let uuid = parse_session_id(session_id)?;
        let txn = self.db.begin().await?;
        let mut record = Sessions::find()
            .filter(Column::Uuid.eq(uuid))
            .lock_exclusive()
            .one(&txn)
            .await?
            .with_context(|| format!("会话[{session_id}]不存在，请核实后重试"))?;

        // 2.只过滤 files，保留其他文件的顺序与元数据；SQL NULL 视为空列表。
        let mut files: Vec<Value> =
            serde_json::from_value(record.files.take().unwrap_or_else(|| json!([])))
                .context("会话 files 必须是 JSON 数组")?;
        let original_length = files.len();
        files.retain(|file| file.get("id").and_then(Value::as_str) != Some(file_id));

        // 3.文件不存在时不重复写入；有变化则只更新文件列，钩子会刷新 updated_at。
        if files.len() != original_length {
            let mut active = record.into_active_model();
            active.files = Set(Some(Value::Array(files)));
            active.update(&txn).await?;
        }

        // 4.显式提交并释放行锁；此前任一步失败，事务丢弃时会回滚。
        txn.commit().await?;
        Ok(())
    }

    /// 根据沙箱路径查询会话中的文件信息。
    async fn get_file_by_path(&self, session_id: &str, filepath: &str) -> Result<Option<File>> {
        // 1.只查询文件列，不加载会话的事件、记忆等大字段。
        let files = Sessions::find()
            .select_only()
            .column(Column::Files)
            .filter(Column::Uuid.eq(parse_session_id(session_id)?))
            .into_tuple::<Option<Value>>()
            .one(&self.db)
            .await?
            .flatten();

        // 2.会话不存在或文件列为 NULL 时返回 None。
        let Some(files) = files else {
            return Ok(None);
        };

        // 3.只转换匹配的文件；损坏的数据需要报错，不能当作查询未命中。
        for file in files.as_array().context("会话 files 必须是 JSON 数组")? {
            if file.get("filepath").and_then(Value::as_str) == Some(filepath) {
                return serde_json::from_value(file.clone())
                    .context("反序列化会话文件失败")
                    .map(Some);
            }
        }
        Ok(None)
    }

    /// 创建或替换某个 Agent 的记忆，保留其他 Agent 的记忆。
    async fn save_memory(&self, session_id: &str, agent_name: &str, memory: Memory) -> Result<()> {
        // 1.记忆先序列化，再构建只有一个 Agent 键的补丁对象。
        let memory_data = serde_json::to_value(memory).context("序列化 Agent 记忆失败")?;
        let patch = json!({ agent_name: memory_data });

        // 2.JSONB 对象在数据库端合并，同名键替换，其他键保留；并非递归合并消息。
        let update = Sessions::update_many().col_expr(
            Column::Memories,
            Expr::cust_with_values("COALESCE(memories, '{}'::jsonb) || $1::jsonb", [patch]),
        );
        // 3.执行原子更新并检查会话是否存在。
        self.update_session(session_id, update).await
    }

    /// 获取指定会话中某个 Agent 的记忆。
    async fn get_memory(&self, session_id: &str, agent_name: &str) -> Result<Memory> {
        // 1.只提取指定键，Agent 名称作为文本参数绑定，特殊字符不影响 SQL。
        let memory = Sessions::find()
            .select_only()
            .expr_as(
                Expr::cust_with_values("memories -> $1::text", [agent_name]),
                "agent_memory",
            )
            .filter(Column::Uuid.eq(parse_session_id(session_id)?))
            .into_tuple::<Option<Value>>()
            .one(&self.db)
            .await?;

        // 2.外层 Option 表示会话是否存在，不能与某个 Agent 尚无记忆混淆。
        let memory = memory.with_context(|| format!("会话[{session_id}]不存在，请核实后重试"))?;

        // 3.缺少该 Agent 的键或值为 null 时返回空记忆，损坏的记忆则返回转换错误。
        match memory {
            None | Some(Value::Null) => Ok(Memory::default()),
            Some(data) => serde_json::from_value(data).context("反序列化 Agent 记忆失败"),
        }
    }
}

/// 将对外的字符串 id 转成数据库 uuid，非法标识作为输入错误向上传播。
fn parse_session_id(session_id: &str) -> Result<Uuid> {
    Uuid::parse_str(session_id).with_context(|| format!("无效的会话 id: {session_id}"))
}
