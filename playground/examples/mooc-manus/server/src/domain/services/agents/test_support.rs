use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Mutex,
    },
};

use anyhow::{bail, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};

use crate::domain::{
    models::{Event, File, Memory, Session, SessionStatus},
    repositories::SessionRepository,
};

/// Agent 与 Flow 单元测试使用的内存仓库。
#[derive(Default)]
pub struct MemoryRepository {
    pub sessions: Mutex<HashMap<String, Session>>,
    pub memories: Mutex<HashMap<(String, String), Memory>>,
    pub reads: AtomicUsize,
    pub writes: AtomicUsize,
    pub fail_read: AtomicBool,
    pub fail_save: AtomicBool,
    pub fail_add_file: AtomicBool,
    pub fail_add_event: AtomicBool,
    pub fail_metadata: AtomicBool,
    pub fail_update_status: AtomicBool,
}

impl MemoryRepository {
    pub fn insert_session(&self, session: Session) {
        self.sessions
            .lock()
            .unwrap()
            .insert(session.id.clone(), session);
    }

    pub fn session(&self, session_id: &str) -> Option<Session> {
        self.sessions.lock().unwrap().get(session_id).cloned()
    }

    pub fn memory(&self, session_id: &str, agent_name: &str) -> Memory {
        self.memories
            .lock()
            .unwrap()
            .get(&(session_id.to_string(), agent_name.to_string()))
            .cloned()
            .unwrap_or_default()
    }

    pub fn insert(&self, session_id: &str, agent_name: &str, memory: Memory) {
        self.memories
            .lock()
            .unwrap()
            .insert((session_id.to_string(), agent_name.to_string()), memory);
    }
}

#[async_trait]
impl SessionRepository for MemoryRepository {
    async fn save(&self, session: Session) -> Result<()> {
        self.insert_session(session);
        Ok(())
    }

    async fn get_all(&self) -> Result<Vec<Session>> {
        bail!("测试中不应调用 get_all")
    }

    async fn get_by_id(&self, session_id: &str) -> Result<Option<Session>> {
        Ok(self.session(session_id))
    }

    async fn delete_by_id(&self, _session_id: &str) -> Result<()> {
        bail!("测试中不应调用 delete_by_id")
    }

    async fn update_title(&self, session_id: &str, title: &str) -> Result<()> {
        if self.fail_metadata.load(Ordering::SeqCst) {
            bail!("模拟更新会话元数据失败");
        }
        let mut sessions = self.sessions.lock().unwrap();
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| anyhow::anyhow!("会话[{session_id}]不存在"))?;
        session.title = title.to_string();
        Ok(())
    }

    async fn update_latest_message(
        &self,
        session_id: &str,
        message: &str,
        timestamp: DateTime<Utc>,
    ) -> Result<()> {
        if self.fail_metadata.load(Ordering::SeqCst) {
            bail!("模拟更新会话元数据失败");
        }
        let mut sessions = self.sessions.lock().unwrap();
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| anyhow::anyhow!("会话[{session_id}]不存在"))?;
        session.latest_message = message.to_string();
        session.latest_message_at = Some(timestamp);
        Ok(())
    }

    async fn update_unread_message_count(&self, _session_id: &str, _count: usize) -> Result<()> {
        bail!("测试中不应调用 update_unread_message_count")
    }

    async fn increment_unread_message_count(&self, session_id: &str) -> Result<()> {
        if self.fail_metadata.load(Ordering::SeqCst) {
            bail!("模拟更新会话元数据失败");
        }
        let mut sessions = self.sessions.lock().unwrap();
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| anyhow::anyhow!("会话[{session_id}]不存在"))?;
        session.unread_message_count += 1;
        Ok(())
    }

    async fn decrement_unread_message_count(&self, _session_id: &str) -> Result<()> {
        bail!("测试中不应调用 decrement_unread_message_count")
    }

    async fn update_status(&self, session_id: &str, status: SessionStatus) -> Result<()> {
        if self.fail_update_status.load(Ordering::SeqCst) {
            bail!("模拟更新会话状态失败");
        }
        let mut sessions = self.sessions.lock().unwrap();
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| anyhow::anyhow!("会话[{session_id}]不存在"))?;
        session.status = status;
        Ok(())
    }

    async fn add_event(&self, session_id: &str, event: Event) -> Result<()> {
        if self.fail_add_event.load(Ordering::SeqCst) {
            bail!("模拟保存事件失败");
        }
        let mut sessions = self.sessions.lock().unwrap();
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| anyhow::anyhow!("会话[{session_id}]不存在"))?;
        session.events.push(event);
        Ok(())
    }

    async fn add_file(&self, session_id: &str, file: File) -> Result<()> {
        if self.fail_add_file.load(Ordering::SeqCst) {
            bail!("模拟添加会话文件失败");
        }
        let mut sessions = self.sessions.lock().unwrap();
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| anyhow::anyhow!("会话[{session_id}]不存在"))?;
        session.files.push(file);
        Ok(())
    }

    async fn remove_file(&self, session_id: &str, file_id: &str) -> Result<()> {
        let mut sessions = self.sessions.lock().unwrap();
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| anyhow::anyhow!("会话[{session_id}]不存在"))?;
        session.files.retain(|file| file.id != file_id);
        Ok(())
    }

    async fn get_file_by_path(&self, session_id: &str, filepath: &str) -> Result<Option<File>> {
        let session = self
            .session(session_id)
            .ok_or_else(|| anyhow::anyhow!("会话[{session_id}]不存在"))?;
        Ok(session
            .files
            .into_iter()
            .find(|file| file.filepath == filepath))
    }

    async fn save_memory(&self, session_id: &str, agent_name: &str, memory: Memory) -> Result<()> {
        self.writes.fetch_add(1, Ordering::SeqCst);
        if self.fail_save.swap(false, Ordering::SeqCst) {
            bail!("模拟保存记忆失败")
        }
        self.insert(session_id, agent_name, memory);
        Ok(())
    }

    async fn get_memory(&self, session_id: &str, agent_name: &str) -> Result<Memory> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        if self.fail_read.swap(false, Ordering::SeqCst) {
            bail!("模拟读取记忆失败")
        }
        Ok(self.memory(session_id, agent_name))
    }
}
