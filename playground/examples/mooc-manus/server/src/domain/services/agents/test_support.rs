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

    async fn update_title(&self, _session_id: &str, _title: &str) -> Result<()> {
        bail!("测试中不应调用 update_title")
    }

    async fn update_latest_message(
        &self,
        _session_id: &str,
        _message: &str,
        _timestamp: DateTime<Utc>,
    ) -> Result<()> {
        bail!("测试中不应调用 update_latest_message")
    }

    async fn update_unread_message_count(&self, _session_id: &str, _count: usize) -> Result<()> {
        bail!("测试中不应调用 update_unread_message_count")
    }

    async fn increment_unread_message_count(&self, _session_id: &str) -> Result<()> {
        bail!("测试中不应调用 increment_unread_message_count")
    }

    async fn decrement_unread_message_count(&self, _session_id: &str) -> Result<()> {
        bail!("测试中不应调用 decrement_unread_message_count")
    }

    async fn update_status(&self, session_id: &str, status: SessionStatus) -> Result<()> {
        let mut sessions = self.sessions.lock().unwrap();
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| anyhow::anyhow!("会话[{session_id}]不存在"))?;
        session.status = status;
        Ok(())
    }

    async fn add_event(&self, _session_id: &str, _event: Event) -> Result<()> {
        bail!("测试中不应调用 add_event")
    }

    async fn add_file(&self, _session_id: &str, _file: File) -> Result<()> {
        bail!("测试中不应调用 add_file")
    }

    async fn remove_file(&self, _session_id: &str, _file_id: &str) -> Result<()> {
        bail!("测试中不应调用 remove_file")
    }

    async fn get_file_by_path(&self, _session_id: &str, _filepath: &str) -> Result<Option<File>> {
        bail!("测试中不应调用 get_file_by_path")
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
