use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::domain::models::{Session, SessionStatus};

use super::{events::AgentSseEvent, files::FileInfoResponse};

/// 聊天请求结构，四个字段均可省略或传入 null。
#[derive(Debug, Deserialize, ToSchema)]
#[serde(default)]
pub struct ChatRequest {
    /// 人类消息。
    pub message: Option<String>,
    /// 附件文件 id 列表，省略时为空列表，显式 null 也按空列表处理。
    pub attachments: Option<Vec<String>>,
    /// 最新事件 id。
    pub event_id: Option<String>,
    /// Unix 时间戳，单位为秒；省略时使用服务端当前时间。
    pub timestamp: Option<i64>,
}

impl Default for ChatRequest {
    fn default() -> Self {
        Self {
            message: None,
            attachments: Some(Vec::new()),
            event_id: None,
            timestamp: None,
        }
    }
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

/// 获取会话详情响应结构。
#[derive(Debug, Serialize, ToSchema)]
pub struct GetSessionResponse {
    /// 会话 id。
    pub session_id: String,
    /// 会话标题。
    pub title: Option<String>,
    /// 会话状态。
    #[schema(value_type = String, example = "pending")]
    pub status: SessionStatus,
    /// 对话过程中产生的事件，空白会话返回空列表。
    pub events: Vec<AgentSseEvent>,
}

impl From<Session> for GetSessionResponse {
    fn from(session: Session) -> Self {
        Self {
            session_id: session.id,
            title: Some(session.title),
            status: session.status,
            events: AgentSseEvent::from_events(session.events),
        }
    }
}

/// 获取会话文件列表响应结构，空白会话返回空列表。
#[derive(Debug, Default, Serialize, ToSchema)]
pub struct GetSessionFilesResponse {
    /// 人类上传与智能体生成的文件信息。
    pub files: Vec<FileInfoResponse>,
}

/// 需要读取的沙箱文件请求结构。
#[derive(Debug, Deserialize, ToSchema)]
pub struct FileReadRequest {
    pub filepath: String,
}

/// 需要读取的沙箱文件响应结构体。
#[derive(Debug, Serialize, ToSchema)]
pub struct FileReadResponse {
    pub filepath: String,
    pub content: String,
}

/// 需要读取的沙箱 Shell 请求结构体。
#[derive(Debug, Deserialize, ToSchema)]
pub struct ShellReadRequest {
    /// Shell 会话 id，与 URL 中的任务会话 id 分别传递。
    pub session_id: String,
}

/// 控制台记录模型，包含 ps1、command、output。
#[derive(Debug, Deserialize, Serialize, ToSchema)]
pub struct ConsoleRecord {
    pub ps1: String,
    pub command: String,
    pub output: String,
}

/// 需要读取的沙箱 Shell 响应结构体。
#[derive(Debug, Deserialize, Serialize, ToSchema)]
pub struct ShellReadResponse {
    pub session_id: String,
    pub output: String,
    #[serde(default)]
    pub console_records: Vec<ConsoleRecord>,
}

/// 操作成功时的可选空对象；清除未读数、删除和停止接口均返回 None。
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

    use super::{
        ChatRequest, EmptySessionData, FileReadResponse, ListSessionItem, SessionResponse,
        ShellReadResponse,
    };
    use crate::domain::models::{Session, SessionStatus};

    #[test]
    fn sandbox_read_responses_preserve_content_and_project_declared_fields() {
        assert_eq!(
            serde_json::to_value(SessionResponse::success(
                "获取会话文件内容成功",
                FileReadResponse {
                    filepath: "/home/ubuntu/示例.txt".into(),
                    content: "第一行\n第二行\n".into(),
                },
            ))
            .unwrap(),
            json!({
                "code": 200, "msg": "获取会话文件内容成功",
                "data": {"filepath": "/home/ubuntu/示例.txt", "content": "第一行\n第二行\n"}
            })
        );
        let data = json!({
            "session_id": "manus-shell", "output": "你好\n",
            "console_records": [
                {"ps1": "ubuntu $", "command": "echo 你好", "output": "你好\n", "extra": true}
            ],
            "extra": "沙箱内部字段"
        });
        let response: ShellReadResponse = serde_json::from_value(data).unwrap();
        assert_eq!(
            serde_json::to_value(response).unwrap(),
            json!({
                "session_id": "manus-shell", "output": "你好\n",
                "console_records": [{"ps1": "ubuntu $", "command": "echo 你好", "output": "你好\n"}]
            })
        );
        let empty: ShellReadResponse =
            serde_json::from_str(r#"{"session_id":"manus-shell","output":""}"#).unwrap();
        assert_eq!(
            serde_json::to_value(empty).unwrap()["console_records"],
            json!([])
        );
    }

    #[test]
    fn malformed_shell_payloads_fail_instead_of_becoming_successful_empty_output() {
        for data in [
            json!({}),
            json!({"session_id": "shell"}),
            json!({"session_id": "shell", "output": null}),
            json!({"session_id": "shell", "output": "", "console_records": null}),
            json!({"session_id": "shell", "output": "", "console_records": [{}]}),
        ] {
            assert!(serde_json::from_value::<ShellReadResponse>(data).is_err());
        }
    }

    #[test]
    fn chat_request_accepts_optional_fields_and_preserves_values() {
        let missing: ChatRequest = serde_json::from_value(json!({})).unwrap();
        assert!(
            missing.message.is_none() && missing.event_id.is_none() && missing.timestamp.is_none()
        );
        assert_eq!(missing.attachments, Some(vec![]));
        let null: ChatRequest = serde_json::from_value(json!({
            "message": null, "attachments": null, "event_id": null, "timestamp": null
        }))
        .unwrap();
        assert!(null.message.is_none() && null.attachments.is_none());
        assert!(null.event_id.is_none() && null.timestamp.is_none());
        let request: ChatRequest = serde_json::from_value(json!({
            "message": " ", "attachments": ["file-1", "file-2"],
            "event_id": "1790784000000-1", "timestamp": 1790784000_i64
        }))
        .unwrap();
        assert_eq!(request.message.as_deref(), Some(" "));
        assert_eq!(request.attachments.unwrap(), vec!["file-1", "file-2"]);
        assert_eq!(request.event_id.as_deref(), Some("1790784000000-1"));
        assert_eq!(request.timestamp, Some(1790784000));
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
