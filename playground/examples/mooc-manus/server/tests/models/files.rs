use chrono::{TimeZone, Utc};
use sea_orm::{
    ActiveValue::{NotSet, Set, Unchanged},
    DatabaseBackend, IntoActiveModel, QueryTrait, TryIntoModel,
};
use server::{
    domain::models::File,
    models::files::{ActiveModel, Entity, Model},
};
use uuid::Uuid;

fn nullable_model() -> Model {
    let timestamp = Utc
        .with_ymd_and_hms(2026, 9, 11, 12, 0, 0)
        .unwrap()
        .fixed_offset();
    Model {
        created_at: timestamp,
        updated_at: timestamp,
        id: 42,
        uuid: Some(Uuid::new_v4()),
        filename: None,
        filepath: None,
        key: None,
        extension: None,
        mime_type: None,
        size: None,
        user_id: None,
        is_deleted: None,
        remark: None,
    }
}

fn file() -> File {
    File {
        filename: "自学笔记.md".to_string(),
        filepath: "/workspace/自学笔记.md".to_string(),
        key: "uploads/2026/09/11/notes.md".to_string(),
        extension: "md".to_string(),
        mime_type: "text/markdown".to_string(),
        size: 1024,
        ..File::default()
    }
}

#[test]
fn converts_nullable_columns_to_domain_defaults_and_uses_uuid() {
    let model = nullable_model();
    let id = model.uuid.unwrap().to_string();
    let file = model.into_file().unwrap();

    assert_eq!(file.id, id);
    assert_eq!(file.filename, "");
    assert_eq!(file.filepath, "");
    assert_eq!(file.key, "");
    assert_eq!(file.extension, "");
    assert_eq!(file.mime_type, "");
    assert_eq!(file.size, 0);
}

#[test]
fn rejects_missing_uuid_and_negative_database_size() {
    let mut model = nullable_model();
    model.uuid = None;
    assert!(model.into_file().unwrap_err().to_string().contains("uuid"));

    let mut model = nullable_model();
    model.size = Some(-1);
    assert!(model.into_file().unwrap_err().to_string().contains("size"));
}

#[test]
fn creates_active_model_with_checked_business_fields_and_database_defaults() {
    let file = file();
    let active = ActiveModel::from_file(file.clone()).unwrap();

    assert_eq!(active.uuid, Set(Some(Uuid::parse_str(&file.id).unwrap())));
    assert_eq!(active.filename, Set(Some(file.filename)));
    assert_eq!(active.filepath, Set(Some(file.filepath)));
    assert_eq!(active.key, Set(Some(file.key)));
    assert_eq!(active.extension, Set(Some(file.extension)));
    assert_eq!(active.mime_type, Set(Some(file.mime_type)));
    assert_eq!(active.size, Set(Some(1024)));
    assert_eq!(active.id, NotSet);
    assert_eq!(active.created_at, NotSet);
    assert_eq!(active.updated_at, NotSet);
    assert_eq!(active.user_id, Set(None));
    assert_eq!(active.is_deleted, Set(Some(false)));
    assert_eq!(active.remark, Set(None));

    let maximum_size = File {
        size: i32::MAX as usize,
        ..File::default()
    };
    assert_eq!(
        ActiveModel::from_file(maximum_size).unwrap().size,
        Set(Some(i32::MAX))
    );
}

#[test]
fn rejects_invalid_uuid_and_size_overflow_before_mutation() {
    for invalid in [
        File {
            id: "invalid-uuid".to_string(),
            ..file()
        },
        File {
            size: i32::MAX as usize + 1,
            ..file()
        },
    ] {
        assert!(ActiveModel::from_file(invalid.clone()).is_err());
        let mut active = nullable_model().into_active_model();
        let before = active.clone();
        assert!(active.apply_file(invalid).is_err());
        assert_eq!(active, before);
    }
}

#[test]
fn applies_all_business_fields_without_overwriting_database_metadata() {
    let mut original = nullable_model();
    original.user_id = Some(Uuid::new_v4());
    original.is_deleted = Some(true);
    original.remark = Some("保留数据库备注".to_string());
    let mut active = original.clone().into_active_model();
    let file = file();

    active.apply_file(file.clone()).unwrap();

    assert_eq!(active.id, Unchanged(original.id));
    assert_eq!(active.created_at, Unchanged(original.created_at));
    assert_eq!(active.updated_at, Unchanged(original.updated_at));
    assert_eq!(active.user_id, Unchanged(original.user_id));
    assert_eq!(active.is_deleted, Unchanged(original.is_deleted));
    assert_eq!(active.remark, Unchanged(original.remark));
    assert_eq!(active.try_into_model().unwrap().into_file().unwrap(), file);
}

#[test]
fn finder_validates_business_uuid_and_selects_uuid_column() {
    let id = Uuid::new_v4();
    let sql = Entity::find_by_uuid(&id.to_string())
        .unwrap()
        .build(DatabaseBackend::Postgres)
        .to_string();
    assert!(sql.contains(&format!("WHERE \"files\".\"uuid\" = '{id}'")));
    assert!(Entity::find_by_uuid("42").is_err());
}
