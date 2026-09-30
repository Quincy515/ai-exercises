use chrono::{DateTime, Utc};
use serde::Serialize;
use utoipa::ToSchema;

use crate::domain::models::{Session, SessionStatus};

/// 创建会话响应结构。
#[derive(Debug, Serialize, ToSchema)]
pub struct CreateSessionResponse {
    /// 会话 id。
    pub session_id: String,
}

/// 会话列表条目基础信息，只投影视图需要的字段。
#[derive(Debug, Serialize, ToSchema)]
pub struct ListSessionItem {
    pub session_id: String,
    pub title: String,
    pub latest_message: String,
    /// 最新消息时间；空白会话还没有消息时为 null。
    #[schema(value_type = Option<String>, format = DateTime)]
    pub latest_message_at: Option<DateTime<Utc>>,
    /// 会话状态：pending、running、waiting、completed。
    #[schema(value_type = String, example = "pending")]
    pub status: SessionStatus,
    pub unread_message_count: usize,
}

impl From<Session> for ListSessionItem {
    fn from(session: Session) -> Self {
        Self {
            session_id: session.id,
            title: session.title,
            latest_message: session.latest_message,
            latest_message_at: session.latest_message_at,
            status: session.status,
            unread_message_count: session.unread_message_count,
        }
    }
}

/// 获取会话列表基础信息响应结构。
#[derive(Debug, Serialize, ToSchema)]
pub struct ListSessionResponse {
    pub sessions: Vec<ListSessionItem>,
}

/// 操作成功时的可选空对象；当前清除未读数和删除接口均返回 None。
#[derive(Debug, Serialize, ToSchema)]
pub struct EmptySessionData {}

/// 会话接口的成功响应；无返回数据的操作使用 data: null。
#[derive(Debug, Serialize, ToSchema)]
pub struct SessionResponse<T> {
    pub code: u16,
    pub msg: String,
    pub data: T,
}

impl<T> SessionResponse<T> {
    pub fn success(msg: impl Into<String>, data: T) -> Self {
        Self {
            code: 200,
            msg: msg.into(),
            data,
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{EmptySessionData, ListSessionItem, SessionResponse};
    use crate::domain::models::{Session, SessionStatus};

    #[test]
    fn list_item_exposes_only_basic_fields_and_keeps_null_time() {
        let session = Session {
            id: "session-1".to_owned(),
            title: "新对话".to_owned(),
            ..Session::default()
        };

        assert_eq!(
            serde_json::to_value(ListSessionItem::from(session)).unwrap(),
            json!({
                "session_id": "session-1",
                "title": "新对话",
                "latest_message": "",
                "latest_message_at": null,
                "status": "pending",
                "unread_message_count": 0
            })
        );
    }

    #[test]
    fn list_item_preserves_domain_status_values() {
        for (status, expected) in [
            (SessionStatus::Pending, "pending"),
            (SessionStatus::Running, "running"),
            (SessionStatus::Waiting, "waiting"),
            (SessionStatus::Completed, "completed"),
        ] {
            let session = Session {
                status,
                ..Session::default()
            };
            let item = serde_json::to_value(ListSessionItem::from(session)).unwrap();
            assert_eq!(item["status"], expected);
        }
    }

    #[test]
    fn success_without_data_serializes_null() {
        let response = SessionResponse::success("清除未读消息数成功", None::<EmptySessionData>);

        assert_eq!(
            serde_json::to_value(response).unwrap(),
            json!({ "code": 200, "msg": "清除未读消息数成功", "data": null })
        );
    }
}
