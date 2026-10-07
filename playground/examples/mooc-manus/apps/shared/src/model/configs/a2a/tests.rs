use crux_core::{App, Command};
use crux_http::{
    HttpError,
    protocol::{HttpRequest, HttpResponse, HttpResult},
};

use crate::{A2aConfigEvent, A2aConfigViewModel, AppCore, ConfigsEvent, Effect, Event, Model};

const BASE: &str = "http://localhost:5150";
const JSON: &str = r#"{"a2a_servers":[{"id":"agent-one","name":"Writer","description":"Writes text","input_modes":["text/plain"],"output_modes":["text/plain"],"streaming":true,"push_notifications":false,"enabled":true}]}"#;
const EMPTY: &str = r#"{"a2a_servers":[]}"#;

fn event(event: A2aConfigEvent) -> Event {
    Event::Configs(ConfigsEvent::A2a(event))
}

fn get() -> Event {
    event(A2aConfigEvent::Get {
        base_url: BASE.to_string(),
    })
}

fn create() -> Event {
    event(A2aConfigEvent::Create {
        base_url: BASE.to_string(),
    })
}

fn edit(url: &str) -> Event {
    event(A2aConfigEvent::EditUrl {
        value: url.to_string(),
    })
}

fn toggle(enabled: bool) -> Event {
    event(A2aConfigEvent::SetEnabled {
        base_url: BASE.to_string(),
        id: "agent-one".to_string(),
        enabled,
    })
}

fn delete() -> Event {
    event(A2aConfigEvent::Delete {
        base_url: BASE.to_string(),
        id: "agent-one".to_string(),
    })
}

fn reply(
    app: &AppCore,
    model: &mut Model,
    mut command: Command<Effect, Event>,
    result: HttpResult,
) -> Command<Effect, Event> {
    command.expect_effect().expect_render();
    command
        .expect_one_effect()
        .expect_http()
        .resolve(result)
        .unwrap();
    app.update(command.expect_one_event(), model)
}

fn success(body: &str) -> HttpResult {
    HttpResult::Ok(HttpResponse::ok().body(body).build())
}

fn loaded(app: &AppCore) -> Model {
    let mut model = Model::default();
    let command = app.update(get(), &mut model);
    reply(app, &mut model, command, success(JSON))
        .expect_one_effect()
        .expect_render();
    model
}

fn state(app: &AppCore, model: &Model) -> A2aConfigViewModel {
    app.view(model).a2a_config
}

#[test]
fn fetches_the_list_contract_and_distinguishes_empty_from_not_loaded() {
    let app = AppCore::default();
    for body in [JSON, EMPTY] {
        let mut model = Model::default();
        assert!(!state(&app, &model).loaded);
        let mut command = app.update(get(), &mut model);
        assert!(state(&app, &model).loading);
        command.expect_effect().expect_render();
        let mut request = command.expect_one_effect().expect_http();
        assert_eq!(
            request.operation,
            HttpRequest::get(format!("{BASE}/api/app_configs/a2a-servers")).build()
        );
        request.resolve(success(body)).unwrap();
        app.update(command.expect_one_event(), &mut model)
            .expect_one_effect()
            .expect_render();
        let state = state(&app, &model);
        assert!(state.loaded);
        assert!(!state.loading);
        assert_eq!(state.servers.len(), usize::from(body == JSON));
        if body == JSON {
            let server = &state.servers[0];
            assert_eq!(server.name, "Writer");
            assert_eq!(server.description, "Writes text");
            assert_eq!(server.input_modes, ["text/plain"]);
            assert_eq!(server.output_modes, ["text/plain"]);
            assert!(server.streaming && server.enabled && !server.push_notifications);
        }
    }
}

#[test]
fn validates_remote_urls_and_normalizes_the_create_body() {
    let app = AppCore::default();
    for value in [
        "",
        " ",
        "invalid",
        "file:///tmp/a2a",
        "ftp://host",
        "https://user:pass@host",
        "https://user@host",
        "https://host?token=x",
        "https://host/#x",
        "https://host/\npath",
    ] {
        let mut model = loaded(&app);
        app.update(edit(value), &mut model)
            .expect_one_effect()
            .expect_render();
        app.update(create(), &mut model)
            .expect_one_effect()
            .expect_render();
        let state = state(&app, &model);
        assert!(state.error.is_some(), "{value}");
        assert_eq!(state.draft_url, value);
        assert!(!state.saving);
    }
    for (value, expected) in [
        (
            "  https://agent.example.test///  ",
            "https://agent.example.test",
        ),
        ("http://localhost:9999/a2a/", "http://localhost:9999/a2a"),
    ] {
        let mut model = loaded(&app);
        app.update(edit(value), &mut model)
            .expect_one_effect()
            .expect_render();
        let mut command = app.update(create(), &mut model);
        command.expect_effect().expect_render();
        let mut request = command.expect_one_effect().expect_http();
        assert_eq!(request.operation.method, "POST");
        assert_eq!(
            request.operation.url,
            format!("{BASE}/api/app_configs/a2a-servers")
        );
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&request.operation.body).unwrap(),
            serde_json::json!({"base_url": expected})
        );
        request.resolve(success("null")).unwrap();
        let mut refresh = app.update(command.expect_one_event(), &mut model);
        let state = state(&app, &model);
        assert!(state.created && state.saving && state.loading);
        assert!(state.draft_url.is_empty());
        assert_eq!(state.notice.as_deref(), Some("已添加远程Agent配置。"));
        refresh.expect_effect().expect_render();
        let request = refresh.expect_one_effect().expect_http();
        assert_eq!(request.operation.method, "GET");
    }
}

#[test]
fn writes_toggle_and_delete_with_null_responses_then_refreshes() {
    let app = AppCore::default();
    for deleting in [false, true] {
        let mut model = loaded(&app);
        let mut command = app.update(if deleting { delete() } else { toggle(false) }, &mut model);
        command.expect_effect().expect_render();
        let mut request = command.expect_one_effect().expect_http();
        let action = if deleting { "delete" } else { "enabled" };
        assert_eq!(
            request.operation.url,
            format!("{BASE}/api/app_configs/a2a-servers/agent-one/{action}")
        );
        assert_eq!(request.operation.method, "POST");
        if deleting {
            assert!(request.operation.body.is_empty());
        } else {
            assert_eq!(
                serde_json::from_slice::<serde_json::Value>(&request.operation.body).unwrap(),
                serde_json::json!({"enabled":false})
            );
        }
        request.resolve(success("null")).unwrap();
        let refresh = app.update(command.expect_one_event(), &mut model);
        let interim = state(&app, &model);
        assert!(interim.saving && interim.loading);
        if deleting {
            assert!(interim.servers.is_empty());
        } else {
            assert!(!interim.servers[0].enabled);
        }
        reply(&app, &mut model, refresh, success(EMPTY))
            .expect_one_effect()
            .expect_render();
        let state = state(&app, &model);
        assert!(!state.saving && !state.loading && state.loaded);
        assert!(state.notice.is_some());
    }
}

#[test]
fn failed_refresh_keeps_confirmed_write_and_never_reposts() {
    let app = AppCore::default();
    let mut model = loaded(&app);
    app.update(edit("https://remote.example.test"), &mut model)
        .expect_one_effect()
        .expect_render();
    let command = app.update(create(), &mut model);
    let refresh = reply(&app, &mut model, command, success("null"));
    reply(
        &app,
        &mut model,
        refresh,
        HttpResult::Err(HttpError::Timeout),
    )
    .expect_one_effect()
    .expect_render();
    let after = state(&app, &model);
    assert!(after.created && after.loaded);
    assert!(!after.saving && !after.loading && !after.write_uncertain);
    assert!(after.error.is_some() && after.notice.is_some());
    assert_eq!(after.servers.len(), 1);
    let mut retry = app.update(get(), &mut model);
    retry.expect_effect().expect_render();
    assert_eq!(
        retry.expect_one_effect().expect_http().operation.method,
        "GET"
    );
}

#[test]
fn guards_busy_reads_writes_and_draft_actions_through_post_refresh() {
    let app = AppCore::default();
    for phase in 0..3 {
        let mut model = loaded(&app);
        app.update(edit("https://remote.example.test"), &mut model)
            .expect_one_effect()
            .expect_render();
        let command = if phase == 0 {
            app.update(get(), &mut model)
        } else {
            app.update(create(), &mut model)
        };
        let _in_flight = if phase == 2 {
            reply(&app, &mut model, command, success("null"))
        } else {
            command
        };
        let before = state(&app, &model);
        for event in [
            get(),
            create(),
            toggle(false),
            delete(),
            edit("http://new.example.test"),
            event(A2aConfigEvent::ResetDraft),
        ] {
            assert!(app.update(event, &mut model).is_done());
            assert_eq!(state(&app, &model), before);
        }
    }
}

#[test]
fn preserves_list_and_draft_after_load_failure_and_allows_retry() {
    let app = AppCore::default();
    for has_data in [false, true] {
        let mut model = if has_data {
            loaded(&app)
        } else {
            Model::default()
        };
        app.update(edit("https://draft.example.test"), &mut model)
            .expect_one_effect()
            .expect_render();
        let previous = state(&app, &model);
        let command = app.update(get(), &mut model);
        reply(
            &app,
            &mut model,
            command,
            HttpResult::Err(HttpError::Io("SECRET".into())),
        )
        .expect_one_effect()
        .expect_render();
        let current = state(&app, &model);
        assert_eq!(current.servers, previous.servers);
        assert_eq!(current.draft_url, previous.draft_url);
        assert_eq!(current.loaded, has_data);
        assert!(!current.error.unwrap().contains("SECRET"));
        let command = app.update(get(), &mut model);
        reply(&app, &mut model, command, success(EMPTY))
            .expect_one_effect()
            .expect_render();
        assert!(state(&app, &model).loaded);
    }
}

#[test]
fn uncertain_writes_require_successful_manual_refresh_before_any_mutation() {
    let app = AppCore::default();
    for failure in [
        HttpResult::Err(HttpError::Timeout),
        HttpResult::Err(HttpError::Io("SECRET".to_string())),
        success("invalid JSON"),
        success(""),
        HttpResult::Ok(HttpResponse::status(500).body("SECRET").build()),
    ] {
        let mut model = loaded(&app);
        app.update(edit("https://remote.example.test"), &mut model)
            .expect_one_effect()
            .expect_render();
        let command = app.update(create(), &mut model);
        reply(&app, &mut model, command, failure)
            .expect_one_effect()
            .expect_render();
        let current = state(&app, &model);
        assert!(current.write_uncertain && !current.saving);
        assert!(current.notice.as_deref().unwrap().contains("避免重复添加"));
        assert!(!current.error.as_deref().unwrap().contains("SECRET"));
        for event in [create(), toggle(false), delete()] {
            assert!(app.update(event, &mut model).is_done());
        }
        let command = app.update(get(), &mut model);
        reply(
            &app,
            &mut model,
            command,
            HttpResult::Err(HttpError::Timeout),
        )
        .expect_one_effect()
        .expect_render();
        assert!(state(&app, &model).write_uncertain);
        assert!(app.update(create(), &mut model).is_done());
        let command = app.update(get(), &mut model);
        reply(&app, &mut model, command, success(JSON))
            .expect_one_effect()
            .expect_render();
        let refreshed = state(&app, &model);
        assert!(!refreshed.write_uncertain);
        assert!(
            refreshed
                .notice
                .as_deref()
                .unwrap()
                .contains("再次添加可能重复")
        );
        assert_eq!(refreshed.draft_url, "https://remote.example.test");
        let mut retry = app.update(create(), &mut model);
        retry.expect_effect().expect_render();
        assert_eq!(
            retry.expect_one_effect().expect_http().operation.method,
            "POST"
        );
    }
}

#[test]
fn known_http_errors_preserve_data_and_allow_manual_retry() {
    let app = AppCore::default();
    for code in [404, 422] {
        let mut model = loaded(&app);
        let before = state(&app, &model).servers;
        let command = app.update(toggle(false), &mut model);
        reply(
            &app,
            &mut model,
            command,
            HttpResult::Ok(HttpResponse::status(code).body("SECRET").build()),
        )
        .expect_one_effect()
        .expect_render();
        let state = state(&app, &model);
        assert_eq!(state.servers, before);
        assert!(!state.saving && !state.write_uncertain);
        assert!(!state.error.as_deref().unwrap().contains("SECRET"));
        if code == 404 {
            assert!(state.error.as_deref().unwrap().contains("刷新列表"));
        }
        assert!(!app.update(toggle(false), &mut model).is_done());
    }
}

#[test]
fn rejects_unknown_resources_unchanged_toggles_and_writes_before_load() {
    let app = AppCore::default();
    let mut model = Model::default();
    for event in [create(), toggle(false), delete()] {
        assert!(app.update(event, &mut model).is_done());
    }
    let mut model = loaded(&app);
    assert!(app.update(toggle(true), &mut model).is_done());
    for event in [
        event(A2aConfigEvent::Delete {
            base_url: BASE.to_string(),
            id: "missing".to_string(),
        }),
        event(A2aConfigEvent::SetEnabled {
            base_url: BASE.to_string(),
            id: "missing".to_string(),
            enabled: false,
        }),
    ] {
        app.update(event, &mut model)
            .expect_one_effect()
            .expect_render();
        assert!(
            state(&app, &model)
                .error
                .as_deref()
                .unwrap()
                .contains("刷新列表")
        );
    }
}

#[test]
fn resets_only_the_draft_and_retains_uncertain_write_guard() {
    let app = AppCore::default();
    let mut model = loaded(&app);
    app.update(edit("https://remote.example.test"), &mut model)
        .expect_one_effect()
        .expect_render();
    let command = app.update(create(), &mut model);
    reply(
        &app,
        &mut model,
        command,
        HttpResult::Err(HttpError::Timeout),
    )
    .expect_one_effect()
    .expect_render();
    let before = state(&app, &model);
    app.update(event(A2aConfigEvent::ResetDraft), &mut model)
        .expect_one_effect()
        .expect_render();
    let state = state(&app, &model);
    assert!(state.draft_url.is_empty() && state.error.is_none());
    assert!(!state.created && state.write_uncertain);
    assert_eq!(state.servers, before.servers);
    assert_eq!(state.notice, before.notice);
}

#[test]
fn preserves_other_config_modules_and_redacts_event_debug() {
    let app = AppCore::default();
    let mut model = loaded(&app);
    let original = app.view(&model);
    app.update(edit("https://remote.example.test"), &mut model)
        .expect_one_effect()
        .expect_render();
    let command = app.update(create(), &mut model);
    reply(
        &app,
        &mut model,
        command,
        HttpResult::Err(HttpError::Io("SECRET".to_string())),
    )
    .expect_one_effect()
    .expect_render();
    let after = app.view(&model);
    assert_eq!(after.agent_config, original.agent_config);
    assert_eq!(after.llm_config, original.llm_config);
    assert_eq!(after.text, original.text);
    for event in [
        event(A2aConfigEvent::MutationFinished(Err(HttpError::Io(
            "SECRET".to_string(),
        )))),
        edit("SECRET"),
    ] {
        assert!(!format!("{event:?}").contains("SECRET"));
    }
}

#[test]
fn rejects_bad_service_urls_and_bad_json_without_losing_existing_data() {
    let app = AppCore::default();
    let mut model = loaded(&app);
    let previous = state(&app, &model).servers;
    app.update(
        event(A2aConfigEvent::Get {
            base_url: "file:///tmp".to_string(),
        }),
        &mut model,
    )
    .expect_one_effect()
    .expect_render();
    assert!(!state(&app, &model).loading);
    app.update(edit("https://remote.example.test"), &mut model)
        .expect_one_effect()
        .expect_render();
    app.update(
        event(A2aConfigEvent::Create {
            base_url: "bad".to_string(),
        }),
        &mut model,
    )
    .expect_one_effect()
    .expect_render();
    assert!(!state(&app, &model).saving);
    let command = app.update(get(), &mut model);
    reply(
        &app,
        &mut model,
        command,
        success(r#"{"a2a_servers":[{}]}"#),
    )
    .expect_one_effect()
    .expect_render();
    assert_eq!(state(&app, &model).servers, previous);
    assert!(state(&app, &model).error.is_some());
}
