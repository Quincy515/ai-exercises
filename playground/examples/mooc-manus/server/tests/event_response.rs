//! 验证流式响应的字段契约；只测试响应投影，不启动数据库、任务或 HTTP 服务。

use chrono::{DateTime, Utc};
use serde_json::{json, Value};
use server::{
    domain::models::{
        BaseEvent, DoneEvent, ErrorEvent, Event, EventType, ExecutionStatus, File, FileToolContent,
        MessageEvent, MessageRole, Plan, PlanEvent, Step, StepEvent, StepEventStatus, TitleEvent,
        ToolContent, ToolEvent, ToolEventStatus, ToolResult, WaitEvent,
    },
    views::events::{AgentSseEvent, BaseEventData, CommonSseEvent, MessageEventData},
};

const SECONDS: i64 = 1_700_000_000;

fn base(event_type: EventType) -> BaseEvent {
    BaseEvent {
        id: "event-1".to_owned(),
        event_type,
        created_at: DateTime::from_timestamp(SECONDS, 987_000_000).unwrap(),
    }
}

#[test]
fn response_uses_unix_seconds_while_domain_keeps_its_timestamp() {
    for seconds in [0, SECONDS] {
        let mut event = base(EventType::Done);
        event.created_at = DateTime::from_timestamp(seconds, 987_000_000).unwrap();
        let original = serde_json::to_value(&event).unwrap();

        assert_eq!(
            serde_json::to_value(BaseEventData::from(&event)).unwrap(),
            json!({"event_id": "event-1", "created_at": seconds})
        );
        assert_eq!(serde_json::to_value(&event).unwrap(), original);
        assert_eq!(original["type"], "done");
        assert_eq!(
            original["created_at"],
            event
                .created_at
                .to_rfc3339_opts(chrono::SecondsFormat::AutoSi, true)
        );
    }
}

#[test]
fn message_preserves_both_roles_and_complete_attachment_metadata() {
    let file = File {
        id: "file-1".to_owned(),
        filename: "报告.txt".to_owned(),
        filepath: "/workspace/报告.txt".to_owned(),
        key: "uploads/file-1".to_owned(),
        extension: ".txt".to_owned(),
        mime_type: "text/plain".to_owned(),
        size: 42,
    };
    for (role, expected_role) in [
        (MessageRole::User, "user"),
        (MessageRole::Assistant, "assistant"),
    ] {
        let event = MessageEvent {
            base: base(EventType::Message),
            role,
            message: "查看这份报告".to_owned(),
            attachments: vec![file.clone()],
        };
        assert_eq!(
            serde_json::to_value(AgentSseEvent::Message(event.into())).unwrap(),
            json!({
                "event": "message",
                "data": {
                    "event_id": "event-1", "created_at": SECONDS,
                    "role": expected_role, "message": "查看这份报告",
                    "attachments": [{
                        "id": "file-1", "filename": "报告.txt", "filepath": "/workspace/报告.txt",
                        "key": "uploads/file-1", "extension": ".txt", "mime_type": "text/plain", "size": 42
                    }]
                }
            })
        );
    }
}

#[test]
fn new_message_data_has_optional_id_current_seconds_and_empty_payload() {
    let before = Utc::now().timestamp();
    let data = MessageEventData::default();
    assert!((before..=Utc::now().timestamp()).contains(&data.base.created_at));
    let created_at = data.base.created_at;
    assert_eq!(
        serde_json::to_value(AgentSseEvent::Message(data)).unwrap(),
        json!({"event": "message", "data": {
            "event_id": null, "created_at": created_at, "role": "assistant",
            "message": "", "attachments": []
        }})
    );
}

#[test]
fn step_uses_nested_step_identity_and_execution_status_without_duplicate_keys() {
    for (status, expected_status) in [
        (ExecutionStatus::Pending, "pending"),
        (ExecutionStatus::Running, "running"),
        (ExecutionStatus::Completed, "completed"),
        (ExecutionStatus::Failed, "failed"),
    ] {
        let event = StepEvent {
            base: base(EventType::Step),
            status: StepEventStatus::Started,
            step: Step {
                id: "step-1".to_owned(),
                description: "读取文件".to_owned(),
                status,
                result: Some("内部执行结果".to_owned()),
                ..Step::default()
            },
        };
        let encoded = serde_json::to_string(&AgentSseEvent::Step(event.into())).unwrap();
        // 直接检查文本，防止重复 JSON 键在解析为 Value 时被覆盖。
        assert_eq!(encoded.matches("\"id\":").count(), 1);
        assert_eq!(
            serde_json::from_str::<Value>(&encoded).unwrap(),
            json!({"event": "step", "data": {
                "event_id": "event-1", "id": "step-1", "created_at": SECONDS,
                "status": expected_status, "description": "读取文件"
            }})
        );
    }
}

#[test]
fn plan_exposes_only_steps_with_individual_ids_and_parent_event_time() {
    let event = PlanEvent {
        base: base(EventType::Plan),
        plan: Plan {
            title: "内部计划标题".to_owned(),
            goal: "内部目标".to_owned(),
            steps: vec![
                Step {
                    id: "step-1".to_owned(),
                    ..Step::new("读取")
                },
                Step {
                    id: "step-2".to_owned(),
                    status: ExecutionStatus::Completed,
                    ..Step::new("汇总")
                },
            ],
            ..Plan::default()
        },
        ..PlanEvent::default()
    };
    let encoded = serde_json::to_string(&AgentSseEvent::Plan(event.into())).unwrap();
    assert_eq!(encoded.matches("\"id\":").count(), 2);
    assert_eq!(encoded.matches("\"event_id\":").count(), 3);
    assert_eq!(
        serde_json::from_str::<Value>(&encoded).unwrap(),
        json!({"event": "plan", "data": {
            "event_id": "event-1", "created_at": SECONDS,
            "steps": [
                {"event_id": "event-1", "id": "step-1", "created_at": SECONDS, "status": "pending", "description": "读取"},
                {"event_id": "event-1", "id": "step-2", "created_at": SECONDS, "status": "completed", "description": "汇总"}
            ]
        }})
    );
}

#[test]
fn tool_renames_fields_and_uses_display_content_instead_of_raw_result() {
    for (status, result, tool_content, expected_status, expected_content) in [
        (ToolEventStatus::Calling, None, None, "calling", Value::Null),
        (
            ToolEventStatus::Called,
            Some(ToolResult {
                success: false,
                message: Some("文件不存在".to_owned()),
                data: Some(json!({"path": "/workspace/a.txt", "lines": []})),
            }),
            Some(ToolContent::File(FileToolContent {
                content: "用于展示的文件正文".to_owned(),
            })),
            "called",
            json!({"content": "用于展示的文件正文"}),
        ),
    ] {
        let event = ToolEvent {
            base: base(EventType::Tool),
            tool_call_id: "call-1".to_owned(),
            tool_name: "file".to_owned(),
            function_name: "file_read".to_owned(),
            function_args: json!({"filepath": "/workspace/a.txt", "options": {"lines": [1, 2]}})
                .as_object()
                .unwrap()
                .clone(),
            function_result: result,
            tool_content,
            status,
        };
        assert_eq!(
            serde_json::to_value(AgentSseEvent::Tool(event.into())).unwrap(),
            json!({"event": "tool", "data": {
                "event_id": "event-1", "created_at": SECONDS, "tool_call_id": "call-1",
                "name": "file", "status": expected_status, "function": "file_read",
                "args": {"filepath": "/workspace/a.txt", "options": {"lines": [1, 2]}},
                "content": expected_content
            }})
        );
    }
}

#[test]
fn unified_mapping_covers_every_domain_variant_and_preserves_list_order() {
    let events = vec![
        Event::Message(MessageEvent {
            base: base(EventType::Message),
            ..MessageEvent::default()
        }),
        Event::Title(TitleEvent {
            base: base(EventType::Title),
            ..TitleEvent::default()
        }),
        Event::Step(StepEvent {
            base: base(EventType::Step),
            ..StepEvent::default()
        }),
        Event::Plan(PlanEvent {
            base: base(EventType::Plan),
            ..PlanEvent::default()
        }),
        Event::Tool(ToolEvent {
            base: base(EventType::Tool),
            ..ToolEvent::default()
        }),
        Event::Done(DoneEvent {
            base: base(EventType::Done),
        }),
        Event::Error(ErrorEvent {
            base: base(EventType::Error),
            ..ErrorEvent::default()
        }),
        Event::Wait(WaitEvent {
            base: base(EventType::Wait),
        }),
    ];
    let responses = AgentSseEvent::from_events(events)
        .into_iter()
        .map(|event| serde_json::to_value(event).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        responses
            .iter()
            .map(|event| event["event"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["message", "title", "step", "plan", "tool", "done", "error", "wait"]
    );
    for event in responses {
        assert_eq!(event["data"]["event_id"], "event-1");
        assert_eq!(event["data"]["created_at"], SECONDS);
        assert!(event["data"].get("type").is_none());
    }
    assert!(AgentSseEvent::from_events(Vec::new()).is_empty());
}

#[test]
fn title_error_done_and_wait_keep_their_expected_data_shape() {
    let cases = [
        (
            AgentSseEvent::Title(
                TitleEvent {
                    base: base(EventType::Title),
                    title: "新标题".to_owned(),
                }
                .into(),
            ),
            json!({"event": "title", "data": {"event_id": "event-1", "created_at": SECONDS, "title": "新标题"}}),
        ),
        (
            AgentSseEvent::Error(
                ErrorEvent {
                    base: base(EventType::Error),
                    error: "执行失败".to_owned(),
                }
                .into(),
            ),
            json!({"event": "error", "data": {"event_id": "event-1", "created_at": SECONDS, "error": "执行失败"}}),
        ),
        (
            AgentSseEvent::Done((&base(EventType::Done)).into()),
            json!({"event": "done", "data": {"event_id": "event-1", "created_at": SECONDS}}),
        ),
        (
            AgentSseEvent::Wait((&base(EventType::Wait)).into()),
            json!({"event": "wait", "data": {"event_id": "event-1", "created_at": SECONDS}}),
        ),
    ];
    for (response, expected) in cases {
        assert_eq!(serde_json::to_value(response).unwrap(), expected);
    }
}

#[test]
fn common_event_preserves_extra_fields_without_adding_another_envelope() {
    let event = Event::Tool(ToolEvent {
        base: base(EventType::Tool),
        tool_call_id: "call-1".to_owned(),
        tool_name: "file".to_owned(),
        function_name: "file_read".to_owned(),
        function_args: json!({"filepath": "a.txt"}).as_object().unwrap().clone(),
        tool_content: Some(ToolContent::File(FileToolContent {
            content: "文件内容".to_owned(),
        })),
        ..ToolEvent::default()
    });
    let response = CommonSseEvent::try_from(&event).unwrap();
    assert_eq!(
        serde_json::to_value(AgentSseEvent::Common(response)).unwrap(),
        json!({"event": "tool", "data": {
            "event_id": "event-1", "created_at": SECONDS, "tool_call_id": "call-1",
            "tool_name": "file", "function_name": "file_read", "function_args": {"filepath": "a.txt"},
            "tool_content": {"content": "文件内容"}, "function_result": null, "status": "calling"
        }})
    );

    // 领域事件类型不一致时，通用转换保留原有序列化错误。
    let invalid = Event::Title(TitleEvent {
        base: base(EventType::Message),
        title: "类型不一致".to_owned(),
    });
    assert!(CommonSseEvent::try_from(&invalid).is_err());
}
