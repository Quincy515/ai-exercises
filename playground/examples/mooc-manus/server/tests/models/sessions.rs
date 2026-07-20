use std::collections::HashMap;

use chrono::{TimeZone, Utc};
use sea_orm::ActiveValue::{NotSet, Set};
use server::{
    domain::models::{Event, File, Memory, Plan, PlanEvent, Session, SessionStatus},
    models::sessions::{ActiveModel, Model},
};
use uuid::Uuid;

#[test]
fn converts_nullable_orm_fields_into_domain_defaults() {
    let uuid = Uuid::new_v4();
    let event = Event::Plan(PlanEvent {
        plan: Plan::new("测试计划", "验证转换"),
        ..PlanEvent::default()
    });
    let file = File {
        filename: "report.md".to_string(),
        ..File::default()
    };
    let memories = HashMap::from([("planner".to_string(), Memory::default())]);
    let timestamp = Utc
        .with_ymd_and_hms(2026, 7, 20, 12, 0, 0)
        .unwrap()
        .fixed_offset();
    let model = Model {
        created_at: timestamp,
        updated_at: timestamp,
        id: 1,
        uuid: Some(uuid),
        sandbox_id: None,
        task_id: None,
        title: None,
        unread_message_count: None,
        latest_message: None,
        latest_message_at: None,
        events: Some(serde_json::to_value(vec![event.clone()]).unwrap()),
        files: Some(serde_json::to_value(vec![file.clone()]).unwrap()),
        memories: Some(serde_json::to_value(&memories).unwrap()),
        status: None,
        user_id: None,
        is_deleted: None,
        remark: None,
    };

    let session = model.into_session().unwrap();

    assert_eq!(session.id, uuid.to_string());
    assert_eq!(session.title, "");
    assert_eq!(session.unread_message_count, 0);
    assert_eq!(session.latest_message, "");
    assert_eq!(session.events, vec![event]);
    assert_eq!(session.files, vec![file]);
    assert_eq!(session.memories, memories);
    assert_eq!(session.status, SessionStatus::Pending);
}

#[test]
fn converts_and_applies_domain_session_to_active_model() {
    let user_id = Uuid::new_v4();
    let session = Session {
        sandbox_id: Some("sandbox-1".to_string()),
        task_id: Some("task-1".to_string()),
        title: "测试会话".to_string(),
        unread_message_count: 2,
        latest_message: "处理中".to_string(),
        events: vec![Event::Plan(PlanEvent {
            plan: Plan::new("执行计划", "完成任务"),
            ..PlanEvent::default()
        })],
        files: vec![File::default()],
        memories: HashMap::from([("executor".to_string(), Memory::default())]),
        status: SessionStatus::Running,
        ..Session::default()
    };

    let active_model = ActiveModel::from_session(session.clone()).unwrap();

    assert_eq!(
        active_model.uuid,
        Set(Some(Uuid::parse_str(&session.id).unwrap()))
    );
    assert_eq!(active_model.title, Set(Some(session.title.clone())));
    assert_eq!(active_model.unread_message_count, Set(Some(2)));
    assert_eq!(
        active_model.events,
        Set(Some(serde_json::to_value(&session.events).unwrap()))
    );
    assert_eq!(
        active_model.status,
        Set(Some(SessionStatus::Running.as_str().to_string()))
    );
    assert_eq!(active_model.created_at, NotSet);
    assert_eq!(active_model.updated_at, NotSet);

    let mut existing = ActiveModel {
        user_id: Set(Some(user_id)),
        remark: Set(Some("保留备注".to_string())),
        ..Default::default()
    };
    existing.apply_session(session).unwrap();

    assert_eq!(existing.user_id, Set(Some(user_id)));
    assert_eq!(existing.remark, Set(Some("保留备注".to_string())));
    assert_eq!(existing.title, Set(Some("测试会话".to_string())));
}
