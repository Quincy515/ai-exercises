use crux_core::{App as _, Command};
use crux_http::{
    HttpError,
    protocol::{HttpRequest, HttpResponse, HttpResult},
};

use crate::{AgentConfigEvent, AppCore, ConfigsEvent, Effect, Event, Model};

use super::{LlmConfigDraft, LlmConfigEvent, LlmConfigField};

const BASE_URL: &str = "http://localhost:5150";
const JSON: &str = r#"{"base_url":"https://provider.example/v1","model_name":"chat-model","temperature":0.7,"max_tokens":8192,"api_key_configured":true}"#;
const SECRET: &str = "test-only-private-key";

fn event(event: LlmConfigEvent) -> Event {
    Event::Configs(ConfigsEvent::Llm(event))
}

fn get() -> Event {
    event(LlmConfigEvent::Get {
        base_url: BASE_URL.to_string(),
    })
}

fn save() -> Event {
    event(LlmConfigEvent::Save {
        base_url: BASE_URL.to_string(),
    })
}

fn edit(field: LlmConfigField, value: &str) -> Event {
    event(LlmConfigEvent::Edit {
        field,
        value: value.to_string(),
    })
}

fn finish_request(
    app: &AppCore,
    model: &mut Model,
    mut command: Command<Effect, Event>,
    response: HttpResult,
) -> HttpRequest {
    command.expect_effect().expect_render();
    let mut request = command.expect_one_effect().expect_http();
    let operation = request.operation.clone();
    request.resolve(response).unwrap();
    let response_event = command.expect_one_event();
    assert!(matches!(
        response_event,
        Event::Configs(ConfigsEvent::Llm(
            LlmConfigEvent::Received(_) | LlmConfigEvent::Saved(_)
        ))
    ));
    app.update(response_event, model)
        .expect_one_effect()
        .expect_render();
    operation
}

fn ok() -> HttpResult {
    HttpResult::Ok(HttpResponse::ok().body(JSON).build())
}

fn loaded_model(app: &AppCore) -> Model {
    let mut model = Model::default();
    let command = app.update(get(), &mut model);
    finish_request(app, &mut model, command, ok());
    model
}

#[test]
fn fetches_endpoint_and_maps_safe_nullable_response() {
    let app = AppCore::default();
    let mut model = Model::default();
    let command = app.update(get(), &mut model);
    assert!(app.view(&model).llm_config.loading);
    assert!(app.update(get(), &mut model).is_done());
    let request = finish_request(&app, &mut model, command, ok());
    assert_eq!(
        request,
        HttpRequest::get("http://localhost:5150/api/app_configs/llm").build()
    );
    let view = app.view(&model).llm_config;
    assert!(!view.loading && !view.dirty && !view.can_save && !view.api_key_changed);
    assert!(view.data.unwrap().api_key_configured);
    assert_eq!(view.draft.temperature, "0.7");
    assert_eq!(view.draft.max_tokens, "8192");
    assert!(view.error.is_none());

    let command = app.update(get(), &mut model);
    finish_request(&app, &mut model, command, HttpResult::Ok(HttpResponse::ok()
        .body(r#"{"base_url":null,"model_name":null,"temperature":null,"max_tokens":null,"api_key_configured":false}"#).build()));
    let view = app.view(&model).llm_config;
    assert_eq!(view.draft, LlmConfigDraft::default());
    assert!(!view.data.unwrap().api_key_configured);
}

#[test]
fn writes_null_for_empty_fields_and_omits_unchanged_api_key() {
    let app = AppCore::default();
    let mut model = loaded_model(&app);
    for field in [
        LlmConfigField::BaseUrl,
        LlmConfigField::ModelName,
        LlmConfigField::Temperature,
        LlmConfigField::MaxTokens,
    ] {
        app.update(edit(field, " "), &mut model)
            .expect_one_effect()
            .expect_render();
    }
    let command = app.update(save(), &mut model);
    let request = finish_request(&app, &mut model, command, ok());
    assert_eq!(request.method, "POST");
    assert_eq!(request.url, "http://localhost:5150/api/app_configs/llm");
    assert!(
        request
            .headers
            .iter()
            .any(|header| header.name.eq_ignore_ascii_case("content-type")
                && header.value.starts_with("application/json"))
    );
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&request.body).unwrap(),
        serde_json::json!({
            "base_url": null, "model_name": null, "temperature": null, "max_tokens": null
        })
    );
    let view = app.view(&model).llm_config;
    // 确认值取服务端响应，即使其结果与本次草稿不同。
    assert_eq!(view.draft.model_name, "chat-model");
    assert!(view.saved && !view.dirty && !view.saving);
}

#[test]
fn serializes_only_a_new_nonempty_key() {
    for api_key in [None, Some(String::new()), Some("  ".to_string())] {
        let mut request = LlmConfigDraft::default().validate("").unwrap();
        request.api_key = api_key;
        let json = serde_json::to_value(request).unwrap();
        assert!(json.get("api_key").is_none());
    }
    let request = LlmConfigDraft::default().validate(SECRET).unwrap();
    assert_eq!(serde_json::to_value(request).unwrap()["api_key"], SECRET);
}

#[test]
fn keeps_secret_out_of_view_debug_and_persistence_and_clears_it_on_success() {
    let app = AppCore::default();
    let mut model = loaded_model(&app);
    let input = edit(LlmConfigField::ApiKey, SECRET);
    assert!(!format!("{input:?}").contains(SECRET));
    app.update(input, &mut model)
        .expect_one_effect()
        .expect_render();
    let view = app.view(&model);
    assert!(view.llm_config.api_key_changed && view.llm_config.dirty && view.llm_config.can_save);
    assert!(!serde_json::to_string(&view).unwrap().contains(SECRET));
    assert!(!format!("{view:?}").contains(SECRET));
    assert!(!serde_json::to_string(&model).unwrap().contains(SECRET));
    let input = event(LlmConfigEvent::Saved(Err(HttpError::Io(
        SECRET.to_string(),
    ))));
    assert!(!format!("{input:?}").contains(SECRET));

    let command = app.update(save(), &mut model);
    let request = finish_request(&app, &mut model, command, ok());
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&request.body).unwrap()["api_key"],
        SECRET
    );
    let view = app.view(&model).llm_config;
    assert!(view.saved && !view.api_key_changed && !view.dirty && !view.can_save);
    assert!(model.configs.llm.api_key.is_empty());
}

#[test]
fn rejects_invalid_url_temperature_and_token_drafts_without_http() {
    let app = AppCore::default();
    for (field, invalid) in [
        (
            LlmConfigField::BaseUrl,
            vec![
                "relative/path",
                "file:///tmp/api",
                "ftp://provider.example",
                "https://user:secret@provider.example",
                "https://user@provider.example",
                "http://",
            ],
        ),
        (
            LlmConfigField::Temperature,
            vec!["NaN", "inf", "-inf", "3", "-2.1", "abc"],
        ),
        (
            LlmConfigField::MaxTokens,
            vec![
                "-1",
                "1.5",
                "1e3",
                "9223372036854775808",
                "18446744073709551616",
                "abc",
            ],
        ),
    ] {
        for value in invalid {
            let mut model = loaded_model(&app);
            let previous = app.view(&model).llm_config.data;
            app.update(edit(field, value), &mut model)
                .expect_one_effect()
                .expect_render();
            app.update(save(), &mut model)
                .expect_one_effect()
                .expect_render();
            let view = app.view(&model).llm_config;
            assert!(view.error.is_some(), "{field:?}: {value}");
            assert!(!view.saving && view.can_save && view.dirty);
            assert_eq!(view.data, previous);
        }
    }
}

#[test]
fn accepts_server_numeric_boundaries_and_trims_public_text() {
    for temperature in ["-2", "2", "0", " 1.5 "] {
        for max_tokens in ["0", "9223372036854775807"] {
            let draft = LlmConfigDraft {
                base_url: " https://provider.example/v1 ".to_string(),
                model_name: " chat-model ".to_string(),
                temperature: temperature.to_string(),
                max_tokens: max_tokens.to_string(),
            };
            let request = draft.validate("").unwrap();
            assert_eq!(
                request.base_url.as_deref(),
                Some("https://provider.example/v1")
            );
            assert_eq!(request.model_name.as_deref(), Some("chat-model"));
            assert_eq!(
                request.temperature,
                Some(temperature.trim().parse().unwrap())
            );
            assert_eq!(request.max_tokens, Some(max_tokens.parse().unwrap()));
        }
    }
}

#[test]
fn failure_preserves_draft_and_key_for_manual_retry_without_exposing_errors() {
    let app = AppCore::default();
    for result in [
        HttpResult::Err(HttpError::Io(SECRET.to_string())),
        HttpResult::Err(HttpError::Timeout),
        HttpResult::Ok(HttpResponse::status(422).body(SECRET).build()),
        HttpResult::Ok(HttpResponse::status(500).body(SECRET).build()),
        HttpResult::Ok(HttpResponse::ok().body(SECRET).build()),
    ] {
        let mut model = loaded_model(&app);
        let previous = app.view(&model).llm_config.data;
        app.update(edit(LlmConfigField::ApiKey, SECRET), &mut model)
            .expect_one_effect()
            .expect_render();
        app.update(edit(LlmConfigField::ModelName, "new-model"), &mut model)
            .expect_one_effect()
            .expect_render();
        let command = app.update(save(), &mut model);
        finish_request(&app, &mut model, command, result);
        let view = app.view(&model).llm_config;
        assert!(!view.error.as_ref().unwrap().contains(SECRET));
        assert_eq!(view.data, previous);
        assert_eq!(view.draft.model_name, "new-model");
        assert!(view.dirty && view.can_save && view.api_key_changed && !view.saving && !view.saved);

        let command = app.update(save(), &mut model);
        assert!(app.view(&model).llm_config.error.is_none());
        let request = finish_request(&app, &mut model, command, ok());
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&request.body).unwrap()["api_key"],
            SECRET
        );
        let view = app.view(&model).llm_config;
        assert!(view.saved && !view.dirty && !view.api_key_changed);
    }
}

#[test]
fn guards_initial_unchanged_and_busy_actions() {
    let app = AppCore::default();
    let mut model = Model::default();
    assert!(app.update(save(), &mut model).is_done());
    assert!(
        app.update(edit(LlmConfigField::ApiKey, SECRET), &mut model)
            .is_done()
    );
    let mut model = loaded_model(&app);
    assert!(app.update(save(), &mut model).is_done());
    for saving in [false, true] {
        let mut model = loaded_model(&app);
        let command = if saving {
            app.update(edit(LlmConfigField::ApiKey, SECRET), &mut model)
                .expect_one_effect()
                .expect_render();
            app.update(save(), &mut model)
        } else {
            app.update(get(), &mut model)
        };
        let before = app.view(&model).llm_config;
        for event in [
            get(),
            save(),
            edit(LlmConfigField::ApiKey, "other"),
            event(LlmConfigEvent::Reset),
        ] {
            assert!(app.update(event, &mut model).is_done());
            assert_eq!(app.view(&model).llm_config, before);
        }
        finish_request(&app, &mut model, command, ok());
    }
}

#[test]
fn dirty_refresh_preserves_draft_and_reset_clears_key_and_errors() {
    let app = AppCore::default();
    let mut model = loaded_model(&app);
    app.update(edit(LlmConfigField::ApiKey, SECRET), &mut model)
        .expect_one_effect()
        .expect_render();
    app.update(edit(LlmConfigField::Temperature, "3"), &mut model)
        .expect_one_effect()
        .expect_render();
    app.update(save(), &mut model)
        .expect_one_effect()
        .expect_render();
    let before = app.view(&model).llm_config;
    assert!(app.update(get(), &mut model).is_done());
    assert_eq!(app.view(&model).llm_config, before);
    app.update(event(LlmConfigEvent::Reset), &mut model)
        .expect_one_effect()
        .expect_render();
    let view = app.view(&model).llm_config;
    assert!(!view.dirty && !view.can_save && !view.api_key_changed);
    assert!(view.error.is_none() && model.configs.llm.api_key.is_empty());
    assert_eq!(view.draft.temperature, "0.7");
}

#[test]
fn agent_and_llm_requests_and_errors_are_isolated() {
    let app = AppCore::default();
    let mut model = loaded_model(&app);
    app.update(edit(LlmConfigField::ApiKey, SECRET), &mut model)
        .expect_one_effect()
        .expect_render();
    let llm_save = app.update(save(), &mut model);
    let mut agent_get = app.update(
        Event::Configs(ConfigsEvent::Agent(AgentConfigEvent::Get {
            base_url: BASE_URL.to_string(),
        })),
        &mut model,
    );
    let view = app.view(&model);
    assert!(view.agent_config.loading && view.llm_config.saving);
    agent_get.expect_effect().expect_render();
    agent_get
        .expect_one_effect()
        .expect_http()
        .resolve(HttpResult::Err(HttpError::Timeout))
        .unwrap();
    app.update(agent_get.expect_one_event(), &mut model)
        .expect_one_effect()
        .expect_render();
    let view = app.view(&model);
    assert!(view.agent_config.error.is_some());
    assert!(view.llm_config.error.is_none() && view.llm_config.saving);
    finish_request(&app, &mut model, llm_save, ok());
    let view = app.view(&model);
    assert!(view.agent_config.error.is_some());
    assert!(view.llm_config.saved && view.llm_config.error.is_none());
}

#[test]
fn rejects_invalid_service_address_without_losing_the_draft() {
    let app = AppCore::default();
    let mut model = Model::default();
    app.update(
        event(LlmConfigEvent::Get {
            base_url: "file:///tmp/api".to_string(),
        }),
        &mut model,
    )
    .expect_one_effect()
    .expect_render();
    assert!(app.view(&model).llm_config.error.is_some());
    assert!(!app.view(&model).llm_config.loading);
    let mut model = loaded_model(&app);
    app.update(edit(LlmConfigField::ApiKey, SECRET), &mut model)
        .expect_one_effect()
        .expect_render();
    app.update(
        event(LlmConfigEvent::Save {
            base_url: "file:///tmp/api".to_string(),
        }),
        &mut model,
    )
    .expect_one_effect()
    .expect_render();
    let view = app.view(&model).llm_config;
    assert!(view.error.is_some() && view.can_save && !view.saving && view.api_key_changed);
}
