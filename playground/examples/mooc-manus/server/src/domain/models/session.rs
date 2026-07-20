use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::domain::models::{Event, File, Memory, Plan};

/// 会话状态类型枚举
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionStatus {
    /// 等待任务
    #[default]
    Pending,
    /// 运行中
    Running,
    /// 等待人类响应
    Waiting,
    /// 已完成
    Completed,
}

impl SessionStatus {
    /// 返回会话状态对应的数据库字符串。
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Waiting => "waiting",
            Self::Completed => "completed",
        }
    }
}

/// 会话领域模型
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Session {
    /// 会话 id
    pub id: String,
    /// 沙箱 id
    pub sandbox_id: Option<String>,
    /// 任务 id
    pub task_id: Option<String>,
    /// 标题
    pub title: String,
    /// 未读消息数
    pub unread_message_count: usize,
    /// 最新消息
    pub latest_message: String,
    /// 最新消息时间
    pub latest_message_at: Option<DateTime<Utc>>,
    /// 事件列表
    pub events: Vec<Event>,
    /// 文件列表
    pub files: Vec<File>,
    /// 记忆
    pub memories: HashMap<String, Memory>,
    /// 状态
    pub status: SessionStatus,
    /// 更新时间
    pub updated_at: DateTime<Utc>,
    /// 创建时间
    pub created_at: DateTime<Utc>,
}

impl Default for Session {
    fn default() -> Self {
        let now = Utc::now();

        Self {
            id: Uuid::new_v4().to_string(),
            sandbox_id: None,
            task_id: None,
            title: String::new(),
            unread_message_count: 0,
            latest_message: String::new(),
            latest_message_at: None,
            events: Vec::new(),
            files: Vec::new(),
            memories: HashMap::new(),
            status: SessionStatus::Pending,
            updated_at: now,
            created_at: now,
        }
    }
}

impl Session {
    /// 获取会话中的最新计划
    pub fn get_latest_plan(&self) -> Option<&Plan> {
        // 1.倒序遍历会话中所有事件消息
        self.events.iter().rev().find_map(|event| {
            // 2.判断事件的类型是否为 PlanEvent，如果是则提取计划后返回
            if let Event::Plan(event) = event {
                Some(&event.plan)
            } else {
                None
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{Session, SessionStatus};
    use crate::domain::models::{Event, Plan, PlanEvent, TitleEvent};
    use uuid::Uuid;

    #[test]
    fn creates_session_with_domain_defaults() {
        let session = Session::default();

        assert!(Uuid::parse_str(&session.id).is_ok());
        assert_eq!(session.status, SessionStatus::Pending);
        assert_eq!(session.unread_message_count, 0);
        assert!(session.events.is_empty());
        assert!(session.files.is_empty());
        assert!(session.memories.is_empty());
    }

    #[test]
    fn returns_latest_plan_from_events() {
        let first_plan = Plan::new("旧计划", "旧目标");
        let latest_plan = Plan::new("新计划", "新目标");
        let session = Session {
            events: vec![
                Event::Plan(PlanEvent {
                    plan: first_plan,
                    ..PlanEvent::default()
                }),
                Event::Title(TitleEvent::default()),
                Event::Plan(PlanEvent {
                    plan: latest_plan.clone(),
                    ..PlanEvent::default()
                }),
            ],
            ..Session::default()
        };

        assert_eq!(session.get_latest_plan(), Some(&latest_plan));
    }
}
