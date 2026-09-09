use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};

use crate::domain::models::{Event, File, Memory, Session, SessionStatus};

/// 会话仓库协议：定义领域层需要的持久化能力，具体存储由基础设施层实现。
///
/// `session_id` 对应领域模型的 `Session.id`，不是数据库的自增主键。
/// 存储或模型转换失败时返回 `Err`；返回 `Option` 的查询用 `Ok(None)` 表达未命中。
///
/// 上层可以通过 trait 对象使用仓库，无需依赖具体数据库：
///
/// ```no_run
/// use anyhow::Result;
/// use server::domain::repositories::SessionRepository;
///
/// async fn load_title(
///     repository: &dyn SessionRepository,
///     session_id: &str,
/// ) -> Result<Option<String>> {
///     Ok(repository.get_by_id(session_id).await?.map(|session| session.title))
/// }
/// ```
#[async_trait]
pub trait SessionRepository: Send + Sync {
    /// 存储会话；相同会话 id 已存在时更新。
    async fn save(&self, session: Session) -> Result<()>;

    /// 获取所有会话；没有会话时返回空列表。
    async fn get_all(&self) -> Result<Vec<Session>>;

    /// 根据会话 id 查询会话；不存在时返回 `Ok(None)`。
    async fn get_by_id(&self, session_id: &str) -> Result<Option<Session>>;

    /// 根据会话 id 删除会话。
    async fn delete_by_id(&self, session_id: &str) -> Result<()>;

    /// 更新会话标题。
    async fn update_title(&self, session_id: &str, title: &str) -> Result<()>;

    /// 同时更新最新消息及其发生时间。
    async fn update_latest_message(
        &self,
        session_id: &str,
        message: &str,
        timestamp: DateTime<Utc>,
    ) -> Result<()>;

    /// 设置未读消息数；类型与领域模型的计数字段一致。
    async fn update_unread_message_count(&self, session_id: &str, count: usize) -> Result<()>;

    /// 将未读消息数增加一条。
    async fn increment_unread_message_count(&self, session_id: &str) -> Result<()>;

    /// 将未读消息数减少一条，最小为零。
    async fn decrement_unread_message_count(&self, session_id: &str) -> Result<()>;

    /// 更新会话状态。
    async fn update_status(&self, session_id: &str, status: SessionStatus) -> Result<()>;

    /// 追加完整事件，保留消息、计划、工具调用等变体的数据。
    async fn add_event(&self, session_id: &str, event: Event) -> Result<()>;

    /// 往会话中新增文件元数据。
    async fn add_file(&self, session_id: &str, file: File) -> Result<()>;

    /// 根据文件 id 移除会话中的文件记录，不负责删除实际文件。
    async fn remove_file(&self, session_id: &str, file_id: &str) -> Result<()>;

    /// 按会话 id 和沙箱内路径查询文件元数据；未找到时返回 `Ok(None)`。
    async fn get_file_by_path(&self, session_id: &str, filepath: &str) -> Result<Option<File>>;

    /// 创建或替换指定 Agent 的记忆，保留其他 Agent 的记忆。
    async fn save_memory(&self, session_id: &str, agent_name: &str, memory: Memory) -> Result<()>;

    /// 获取指定 Agent 的记忆；会话存在但尚无该 Agent 的记忆时返回空记忆。
    /// 会话不存在或读取失败时返回 `Err`。
    async fn get_memory(&self, session_id: &str, agent_name: &str) -> Result<Memory>;
}
