use loco_rs::testing::prelude::*;
use sea_orm::{
    ActiveModelTrait, ActiveValue::Set, ColumnTrait, EntityTrait, IntoActiveModel, QueryFilter,
};
use serial_test::serial;
use server::{
    app::App,
    domain::{
        models::{A2aConfig, A2aServerConfig, AppConfig},
        repositories::AppConfigRepository,
    },
    infrastructure::repositories::SeaOrmAppConfigRepository,
    models::a2a_servers::{A2aServers, Column},
};
use uuid::Uuid;

#[tokio::test]
#[serial]
async fn persists_a2a_config_through_model_facade() {
    let boot = boot_test::<App>().await.unwrap();
    let db = &boot.app_context.db;
    let id = Uuid::new_v4().to_string();
    let uuid = Uuid::parse_str(&id).unwrap();
    let initial = A2aConfig {
        a2a_servers: vec![A2aServerConfig {
            id: id.clone(),
            base_url: "https://agent.example.com".to_string(),
            enabled: true,
        }],
    };

    A2aServers::save_a2a_config(db, initial.clone())
        .await
        .unwrap();
    assert_eq!(
        A2aServers::load_a2a_config(db).await.unwrap(),
        Some(initial)
    );

    let model = A2aServers::find()
        .filter(Column::Uuid.eq(uuid))
        .one(db)
        .await
        .unwrap()
        .unwrap();
    let original_row_id = model.id;
    let agent_card = serde_json::json!({"name": "remote-agent"});
    let mut active_model = model.into_active_model();
    active_model.agent_card = Set(Some(agent_card.clone()));
    active_model.update(db).await.unwrap();

    let updated = A2aConfig {
        a2a_servers: vec![A2aServerConfig {
            id: id.clone(),
            base_url: "https://updated-agent.example.com".to_string(),
            enabled: false,
        }],
    };
    A2aServers::save_a2a_config(db, updated.clone())
        .await
        .unwrap();
    assert_eq!(
        A2aServers::load_a2a_config(db).await.unwrap(),
        Some(updated.clone())
    );

    A2aServers::save_a2a_config(db, A2aConfig::default())
        .await
        .unwrap();
    assert_eq!(A2aServers::load_a2a_config(db).await.unwrap(), None);

    A2aServers::save_a2a_config(db, updated.clone())
        .await
        .unwrap();

    let models = A2aServers::find()
        .filter(Column::Uuid.eq(uuid))
        .all(db)
        .await
        .unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].id, original_row_id);
    assert_eq!(models[0].agent_card.as_ref(), Some(&agent_card));
    assert_eq!(models[0].is_deleted, Some(false));
    assert_eq!(
        A2aServers::load_a2a_config(db).await.unwrap(),
        Some(updated.clone())
    );

    let duplicate_config = A2aConfig {
        a2a_servers: vec![
            updated.a2a_servers[0].clone(),
            updated.a2a_servers[0].clone(),
        ],
    };
    let err = A2aServers::save_a2a_config(db, duplicate_config)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("duplicate A2A server id"));

    let models = A2aServers::find()
        .filter(Column::Uuid.eq(uuid))
        .all(db)
        .await
        .unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].id, original_row_id);
}

#[tokio::test]
#[serial]
async fn app_config_repository_persists_a2a_config() {
    let boot = boot_test::<App>().await.unwrap();
    let repository = SeaOrmAppConfigRepository::new(boot.app_context.db.clone());
    let expected = AppConfig {
        a2a_config: A2aConfig {
            a2a_servers: vec![A2aServerConfig {
                id: Uuid::new_v4().to_string(),
                base_url: "https://agent.example.com".to_string(),
                enabled: true,
            }],
        },
        ..AppConfig::default()
    };

    repository.save(expected.clone()).await.unwrap();

    assert_eq!(repository.load().await.unwrap(), Some(expected));
}
