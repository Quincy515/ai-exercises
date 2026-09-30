use std::sync::Arc;

use anyhow::Result;

use crate::domain::{models::Session, repositories::SessionRepository};

/// 删除会话时，指定的会话不存在。
#[derive(Debug, thiserror::Error)]
#[error("会话[{session_id}]不存在, 删除失败")]
pub struct SessionNotFound {
    pub session_id: String,
}

/// 会话服务：通过仓库编排会话的创建、查询、未读数更新与删除。
pub struct SessionService {
    session_repository: Arc<dyn SessionRepository>,
}

impl SessionService {
    /// 构造函数，完成会话服务初始化。
    pub fn new(session_repository: Arc<dyn SessionRepository>) -> Self {
        Self { session_repository }
    }

    /// 创建一个空白的新任务会话。
    pub async fn create_session(&self) -> Result<Session> {
        tracing::info!("创建一个空白新任务会话");
        let session = Session {
            title: "新对话".to_owned(),
            ..Session::default()
        };
        self.session_repository.save(session.clone()).await?;
        tracing::info!(session_id = %session.id, "成功创建一个新任务会话");
        Ok(session)
    }

    /// 获取项目所有任务会话列表。
    pub async fn get_all_sessions(&self) -> Result<Vec<Session>> {
        self.session_repository.get_all().await
    }

    /// 清空指定会话未读消息数。
    pub async fn clear_unread_message_count(&self, session_id: &str) -> Result<()> {
        tracing::info!(session_id, "清除会话未读消息数");
        self.session_repository
            .update_unread_message_count(session_id, 0)
            .await
    }

    /// 根据传递的会话 id 删除任务会话。
    pub async fn delete_session(&self, session_id: &str) -> Result<()> {
        // 1.先检查会话是否存在。
        tracing::info!(session_id, "正在删除会话");
        if self
            .session_repository
            .get_by_id(session_id)
            .await?
            .is_none()
        {
            tracing::error!(session_id, "会话不存在, 删除失败");
            return Err(SessionNotFound {
                session_id: session_id.to_owned(),
            }
            .into());
        }

        // 2.根据传递的会话 id 删除会话。
        self.session_repository.delete_by_id(session_id).await?;
        tracing::info!(session_id, "删除会话成功");
        Ok(())
    }
}
