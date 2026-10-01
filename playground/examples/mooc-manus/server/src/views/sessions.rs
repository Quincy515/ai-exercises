use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::domain::models::{Session, SessionStatus};

/// 聊天请求结构，四个字段均可省略或传入 null。
#[derive(Debug, Default, Deserialize, ToSchema)]
pub struct ChatRequest {
    /// 人类消息。
    pub message: Option<String>,
    /// 附件列表。
    pub attachments: Option<Vec<String>>,
    /// 最新事件 id。
    pub event_id: Option<String>,
    /// 当前时间戳，本课暂不处理其单位或转换。
    pub timestamp: Option<i64>,
}

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

    use super::{ChatRequest, EmptySessionData, ListSessionItem, SessionResponse};
    use crate::domain::models::{Session, SessionStatus};

    #[test]
    fn chat_request_accepts_optional_fields_and_preserves_values() {
        for value in [
            json!({}),
            json!({"message": null, "attachments": null, "event_id": null, "timestamp": null}),
        ] {
            let request: ChatRequest = serde_json::from_value(value).unwrap();
            assert!(request.message.is_none() && request.attachments.is_none());
            assert!(request.event_id.is_none() && request.timestamp.is_none());
        }
        let request: ChatRequest = serde_json::from_value(json!({
            "message": " ", "attachments": ["file-1", "file-2"],
            "event_id": "1790784000000-1", "timestamp": 1790784000000_i64
        }))
        .unwrap();
        assert_eq!(request.message.as_deref(), Some(" "));
        assert_eq!(request.attachments.unwrap(), vec!["file-1", "file-2"]);
        assert_eq!(request.event_id.as_deref(), Some("1790784000000-1"));
        assert_eq!(request.timestamp, Some(1790784000000));
    }

    #[test]
    fn chat_request_rejects_wrong_field_types() {
        for value in [
            json!({"message": 1}),
            json!({"attachments": "file-1"}),
            json!({"event_id": []}),
            json!({"timestamp": 1.5}),
        ] {
            assert!(serde_json::from_value::<ChatRequest>(value).is_err());
        }
    }

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
