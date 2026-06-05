use loco_rs::testing::prelude::*;
use serial_test::serial;
use server::{
    app::App,
    domain::{
        models::{AgentConfig, AppConfig, LlmConfig},
        repositories::AppConfigRepository,
    },
    infrastructure::repositories::SeaOrmAppConfigRepository,
    models::{agent_configs::AgentConfigs, llm_configs::LlmConfigs},
};

#[tokio::test]
#[serial]
async fn persists_agent_config_through_model_facade() {
    let boot = boot_test::<App>().await.unwrap();
    let db = &boot.app_context.db;
    let initial = AgentConfig {
        max_iterations: 120,
        max_retries: 4,
        max_search_results: 18,
    };

    AgentConfigs::save_agent_config(db, initial.clone())
        .await
        .unwrap();
    assert_eq!(
        AgentConfigs::load_agent_config(db).await.unwrap(),
        Some(initial)
    );

    let updated = AgentConfig {
        max_iterations: 240,
        max_retries: 6,
        max_search_results: 24,
    };
    AgentConfigs::save_agent_config(db, updated.clone())
        .await
        .unwrap();
    assert_eq!(
        AgentConfigs::load_agent_config(db).await.unwrap(),
        Some(updated)
    );
}

#[tokio::test]
#[serial]
async fn app_config_repository_persists_llm_and_agent_configs() {
    let boot = boot_test::<App>().await.unwrap();
    let repository = SeaOrmAppConfigRepository::new(boot.app_context.db.clone());
    let expected = AppConfig {
        llm_config: LlmConfig {
            api_key: Some("test-key".to_string()),
            ..LlmConfig::default()
        },
        agent_config: AgentConfig {
            max_iterations: 320,
            max_retries: 5,
            max_search_results: 16,
        },
        ..AppConfig::default()
    };

    repository.save(expected.clone()).await.unwrap();

    assert_eq!(repository.load().await.unwrap(), Some(expected));
}

#[tokio::test]
#[serial]
async fn app_config_repository_loads_legacy_llm_row_with_default_agent_config() {
    let boot = boot_test::<App>().await.unwrap();
    let db = &boot.app_context.db;
    let llm_config = LlmConfig {
        model_name: Some("legacy-model".to_string()),
        ..LlmConfig::default()
    };

    LlmConfigs::save_llm_config(db, llm_config.clone())
        .await
        .unwrap();

    let repository = SeaOrmAppConfigRepository::new(db.clone());
    assert_eq!(
        repository.load().await.unwrap(),
        Some(AppConfig {
            llm_config,
            agent_config: AgentConfig::default(),
            ..AppConfig::default()
        })
    );
}
