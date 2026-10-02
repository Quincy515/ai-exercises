use std::sync::Arc;

use anyhow::{anyhow, Context, Result};

use crate::domain::{
    external::{Sandbox, SandboxFactory},
    models::{File, Session},
    repositories::SessionRepository,
};

/// 删除会话时，指定的会话不存在。
#[derive(Debug, thiserror::Error)]
#[error("会话[{session_id}]不存在, 删除失败")]
pub struct SessionNotFound {
    pub session_id: String,
}

/// 会话沙箱读取中的业务错误，由控制器映射为 HTTP 状态。
#[derive(Debug, thiserror::Error)]
pub enum SessionSandboxError {
    #[error("当前会话无沙箱环境")]
    Unassigned,
    #[error("当前会话沙箱不存在或已销毁")]
    Unavailable,
    #[error("{0}")]
    RequestFailed(String),
}

/// 会话服务：编排会话管理以及关联沙箱的内容读取。
pub struct SessionService {
    session_repository: Arc<dyn SessionRepository>,
    sandbox_factory: Arc<dyn SandboxFactory>,
}

impl SessionService {
    /// 构造函数，完成会话服务初始化。
    pub fn new(
        session_repository: Arc<dyn SessionRepository>,
        sandbox_factory: Arc<dyn SandboxFactory>,
    ) -> Self {
        Self {
            session_repository,
            sandbox_factory,
        }
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

    /// 获取指定会话详情信息。
    pub async fn get_session(&self, session_id: &str) -> Result<Option<Session>> {
        self.session_repository.get_by_id(session_id).await
    }

    /// 根据传递的会话 id 获取指定会话的文件列表信息。
    pub async fn get_session_files(&self, session_id: &str) -> Result<Vec<File>> {
        tracing::info!(session_id, "获取指定会话下的文件列表信息");
        let session = self
            .session_repository
            .get_by_id(session_id)
            .await?
            .ok_or_else(|| anyhow!("当前会话不存在[{session_id}], 请核实后重试"))?;
        Ok(session.files)
    }

    /// 根据传递的信息查看会话中指定文件的内容。
    pub async fn read_file(&self, session_id: &str, filepath: &str) -> Result<String> {
        tracing::info!(session_id, filepath, "获取会话中的文件内容");
        // 1.检查会话是否存在；2.根据沙箱 id 获取沙箱并判断是否存在。
        let sandbox = self.get_session_sandbox(session_id).await?;

        // 3.调用沙箱读取文件内容，沿用起止行号、权限和最大长度的默认值。
        let result = sandbox.read_file(filepath, None, None, None, None).await?;
        if result.success {
            return result.data.context("沙箱文件读取成功响应缺少 data");
        }
        Err(SessionSandboxError::RequestFailed(result.message.unwrap_or_default()).into())
    }

    /// 根据传递的任务会话 id + Shell 会话 id 获取 Shell 执行结果。
    pub async fn read_shell_output(
        &self,
        session_id: &str,
        shell_session_id: &str,
    ) -> Result<String> {
        tracing::info!(session_id, shell_session_id, "获取会话中的 Shell 内容输出");
        // 1.检查会话是否存在；2.根据沙箱 id 获取沙箱并判断是否存在。
        let sandbox = self.get_session_sandbox(session_id).await?;

        // 3.调用沙箱查看 Shell 内容，包含控制台记录。
        let result = sandbox
            .read_shell_output(shell_session_id, Some(true))
            .await?;
        if result.success {
            // 现有沙箱协议保留完整 JSON 文本，由控制器解析为响应结构。
            return result.data.context("沙箱 Shell 读取成功响应缺少 data");
        }
        Err(SessionSandboxError::RequestFailed(result.message.unwrap_or_default()).into())
    }

    async fn get_session_sandbox(&self, session_id: &str) -> Result<Box<dyn Sandbox>> {
        // 1.检查会话是否存在，查询完成后即归还数据库连接。
        let session = self
            .session_repository
            .get_by_id(session_id)
            .await?
            .ok_or_else(|| anyhow!("当前会话不存在[{session_id}], 请核实后重试"))?;

        // 2.根据沙箱 id 获取沙箱并判断是否存在；读取操作只查找已有沙箱。
        let sandbox_id = session
            .sandbox_id
            .as_deref()
            .filter(|id| !id.is_empty())
            .ok_or(SessionSandboxError::Unassigned)?;
        self.sandbox_factory
            .get(sandbox_id)
            .await?
            .ok_or_else(|| SessionSandboxError::Unavailable.into())
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
