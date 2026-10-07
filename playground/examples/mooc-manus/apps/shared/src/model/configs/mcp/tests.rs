use crux_core::{App, Command};
use crux_http::{
    HttpError,
    protocol::{HttpRequest, HttpResponse, HttpResult},
};
use serde_json::{Value, json};

use super::validation::parse_config;
use crate::{AppCore, ConfigsEvent, Effect, Event, McpConfigEvent, McpConfigViewModel, Model};

const BASE: &str = "http://localhost:5150";
const LIST: &str = r#"{"mcp_servers":[{"server_name":"demo","enabled":true,"transport":"stdio","tools":["search"]}]}"#;
const EMPTY: &str = r#"{"mcp_servers":[]}"#;
const CONFIG: &str = r#"{"mcpServers":{"demo":{"transport":"stdio","command":"test-only-command","args":["TEST_SECRET"],"env":{"TOKEN":"TEST_SECRET"},"headers":{"Authorization":"TEST_SECRET"},"enabled":true}}}"#;
const ACK: &str = r#"{"mcpServers":{"demo":{"transport":"stdio","enabled":true,"command":"TEST_SECRET","args":["TEST_SECRET"],"env":{"TOKEN":"TEST_SECRET"},"headers":{"Authorization":"TEST_SECRET"},"url":"https://host?token=TEST_SECRET"}}}"#;
const DISABLED: &str = r#"{"mcpServers":{"demo":{"transport":"stdio","enabled":false}}}"#;
const DELETED: &str = r#"{"mcpServers":{}}"#;

fn event(event: McpConfigEvent) -> Event {
    Event::Configs(ConfigsEvent::Mcp(event))
}
fn get() -> Event {
    event(McpConfigEvent::Get {
        base_url: BASE.to_string(),
    })
}
fn create() -> Event {
    event(McpConfigEvent::Create {
        base_url: BASE.to_string(),
    })
}
fn edit(value: &str) -> Event {
    event(McpConfigEvent::EditJson {
        value: value.to_string(),
    })
}
fn toggle(enabled: bool) -> Event {
    event(McpConfigEvent::SetEnabled {
        base_url: BASE.to_string(),
        server_name: "demo".to_string(),
        enabled,
    })
}
fn delete() -> Event {
    event(McpConfigEvent::Delete {
        base_url: BASE.to_string(),
        server_name: "demo".to_string(),
    })
}
fn success(body: &str) -> HttpResult {
    HttpResult::Ok(HttpResponse::ok().body(body).build())
}
fn state(app: &AppCore, model: &Model) -> McpConfigViewModel {
    app.view(model).mcp_config
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
fn loaded(app: &AppCore) -> Model {
    let mut model = Model::default();
    let command = app.update(get(), &mut model);
    reply(app, &mut model, command, success(LIST))
        .expect_one_effect()
        .expect_render();
    model
}
fn draft(app: &AppCore, model: &mut Model) {
    app.update(edit(CONFIG), model)
        .expect_one_effect()
        .expect_render();
}

#[test]
fn fetches_list_contract_and_distinguishes_empty_from_initial_state() {
    let app = AppCore::default();
    for body in [LIST, EMPTY] {
        let mut model = Model::default();
        assert!(!state(&app, &model).loaded);
        let mut command = app.update(get(), &mut model);
        assert!(state(&app, &model).loading);
        command.expect_effect().expect_render();
        let mut request = command.expect_one_effect().expect_http();
        assert_eq!(
            request.operation,
            HttpRequest::get(format!("{BASE}/api/app_configs/mcp-servers")).build()
        );
        request.resolve(success(body)).unwrap();
        app.update(command.expect_one_event(), &mut model)
            .expect_one_effect()
            .expect_render();
        let current = state(&app, &model);
        assert!(current.loaded && !current.loading);
        assert_eq!(current.servers.len(), usize::from(body == LIST));
        if body == LIST {
            assert_eq!(current.servers[0].tools, ["search"]);
        }
    }
}

#[test]
fn accepts_batch_defaults_and_nullable_optional_fields_without_copying_secrets() {
    let input = json!({"mcpServers":{
        "one":{"url":"https://mcp.example.test/path?token=TEST_SECRET","env":null,"headers":null,"args":null,"description":null,"command":null},
        "two":{"transport":"stdio","command":"node","args":["TEST_SECRET"],"enabled":false,"env":{"KEY":"TEST_SECRET"},"headers":{"Authorization":"Bearer TEST_SECRET"}}
    }});
    let (request, expected) = parse_config(&input.to_string()).unwrap();
    assert_eq!(expected["one"].transport, "streamable_http");
    assert!(expected["one"].enabled);
    assert!(!expected["two"].enabled);
    assert!(!format!("{expected:?}").contains("TEST_SECRET"));
    let body = serde_json::to_value(request).unwrap();
    assert_eq!(body["mcpServers"]["one"]["env"], Value::Null);
    assert_eq!(body["mcpServers"]["one"]["transport"], "streamable_http");
    assert_eq!(body["mcpServers"]["two"]["args"][0], "TEST_SECRET");
    assert!(body["mcpServers"]["two"].get("description").is_none());
}

#[test]
fn rejects_invalid_json_shapes_unknown_fields_transports_and_runtime_values() {
    let mut invalid = vec![
        "".to_string(),
        "invalid TEST_SECRET".to_string(),
        "null".to_string(),
        "[]".to_string(),
        "{}".to_string(),
        r#"{"mcpServers":{}}"#.to_string(),
        r#"{"mcp_servers":{}}"#.to_string(),
        r#"{"mcpServers":{},"other":"TEST_SECRET"}"#.to_string(),
    ];
    for config in [
        Value::Null,
        json!([]),
        json!({}),
        json!({"transport":"sse","url":"https://host"}),
        json!({"transport":null,"url":"https://host"}),
        json!({"enabled":1,"url":"https://host"}),
        json!({"disabled":true,"url":"https://host"}),
        json!({"type":"stdio","command":"node"}),
        json!({"transport":"stdio","command":" "}),
        json!({"transport":"stdio","command":"node\0"}),
        json!({"transport":"stdio","command":12}),
        json!({"url":"file:///tmp/mcp"}),
        json!({"url":"https://user:pass@host"}),
        json!({"url":"https://host/#fragment"}),
        json!({"url":"https://host/\npath"}),
        json!({"url":"https://host","args":[1]}),
        json!({"url":"https://host","description":3}),
        json!({"url":"https://host","env":[]}),
        json!({"url":"https://host","env":{"TOKEN":2}}),
        json!({"url":"https://host","env":{"BAD=KEY":"value"}}),
        json!({"url":"https://host","env":{"TOKEN":"\0"}}),
        json!({"url":"https://host","headers":{"Authorization":false}}),
        json!({"url":"https://host","headers":{"Bad Key":"value"}}),
        json!({"url":"https://host","headers":{"Authorization":"TEST_SECRET\r\nInjected: value"}}),
    ] {
        invalid.push(json!({"mcpServers":{"demo":config}}).to_string());
    }
    for name in ["", " ", ".", "..", "name\n"] {
        invalid.push(json!({"mcpServers":{name:{"url":"https://host"}}}).to_string());
    }
    let app = AppCore::default();
    for input in invalid {
        let mut model = loaded(&app);
        app.update(edit(&input), &mut model)
            .expect_one_effect()
            .expect_render();
        app.update(create(), &mut model)
            .expect_one_effect()
            .expect_render();
        let current = state(&app, &model);
        assert!(current.error.is_some(), "{input}");
        assert!(!current.error.unwrap().contains("TEST_SECRET"));
        assert!(!current.saving);
    }
}

#[test]
fn sends_json_upsert_and_reconciles_safe_confirmation_before_refresh() {
    let app = AppCore::default();
    let mut model = loaded(&app);
    draft(&app, &mut model);
    let before = app.view(&model);
    assert!(
        !serde_json::to_string(&before)
            .unwrap()
            .contains("TEST_SECRET")
    );
    assert!(
        !serde_json::to_string(&model)
            .unwrap()
            .contains("TEST_SECRET")
    );
    assert!(!format!("{:?}", edit(CONFIG)).contains("TEST_SECRET"));
    let mut command = app.update(create(), &mut model);
    command.expect_effect().expect_render();
    let mut request = command.expect_one_effect().expect_http();
    assert_eq!(request.operation.method, "POST");
    assert_eq!(
        request.operation.url,
        format!("{BASE}/api/app_configs/mcp-servers")
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&request.operation.body).unwrap(),
        serde_json::from_str::<Value>(CONFIG).unwrap()
    );
    request.resolve(success(ACK)).unwrap();
    let refresh = app.update(command.expect_one_event(), &mut model);
    let current = state(&app, &model);
    assert!(current.saving && current.loading && current.created);
    assert!(!current.draft_present);
    assert!(current.servers[0].tools.is_empty());
    assert!(!format!("{current:?}").contains("TEST_SECRET"));
    assert_eq!(current.notice.as_deref(), Some("已保存 MCP 服务器配置。"));
    reply(&app, &mut model, refresh, success(LIST))
        .expect_one_effect()
        .expect_render();
    let after = app.view(&model);
    assert!(!after.mcp_config.saving && !after.mcp_config.loading);
    assert_eq!(after.agent_config, before.agent_config);
    assert_eq!(after.llm_config, before.llm_config);
    assert_eq!(after.a2a_config, before.a2a_config);
}

#[test]
fn batch_create_accepts_same_name_update_and_new_confirmed_rows() {
    let app = AppCore::default();
    let mut model = loaded(&app);
    let input = r#"{"mcpServers":{"demo":{"transport":"stdio","command":"new-command"},"new":{"transport":"stdio","command":"node"}}}"#;
    app.update(edit(input), &mut model)
        .expect_one_effect()
        .expect_render();
    let command = app.update(create(), &mut model);
    let ack = r#"{"mcpServers":{"demo":{"transport":"stdio","enabled":true},"new":{"transport":"stdio","enabled":true}}}"#;
    let refresh = reply(&app, &mut model, command, success(ack));
    let current = state(&app, &model);
    assert_eq!(current.servers.len(), 2);
    assert!(current.servers.iter().all(|server| server.tools.is_empty()));
    assert_eq!(current.servers[0].server_name, "demo");
    assert!(current.servers[0].enabled);
    reply(
        &app,
        &mut model,
        refresh,
        HttpResult::Err(HttpError::Timeout),
    )
    .expect_one_effect()
    .expect_render();
    let current = state(&app, &model);
    assert!(current.created && current.error.is_some() && current.notice.is_some());
    assert!(!current.saving && !current.write_uncertain);
    assert_eq!(current.servers.len(), 2);
    let mut retry = app.update(get(), &mut model);
    retry.expect_effect().expect_render();
    assert_eq!(
        retry.expect_one_effect().expect_http().operation.method,
        "GET"
    );
}

#[test]
fn toggle_and_delete_confirm_results_then_get_without_exposing_response_secrets() {
    let app = AppCore::default();
    for deleting in [false, true] {
        let mut model = loaded(&app);
        let mut command = app.update(if deleting { delete() } else { toggle(false) }, &mut model);
        command.expect_effect().expect_render();
        let mut request = command.expect_one_effect().expect_http();
        let action = if deleting { "delete" } else { "enabled" };
        assert_eq!(
            request.operation.url,
            format!("{BASE}/api/app_configs/mcp-servers/demo/{action}")
        );
        assert_eq!(request.operation.method, "POST");
        if deleting {
            assert!(request.operation.body.is_empty());
        } else {
            assert_eq!(
                serde_json::from_slice::<Value>(&request.operation.body).unwrap(),
                json!({"enabled":false})
            );
        }
        request
            .resolve(success(if deleting { DELETED } else { DISABLED }))
            .unwrap();
        let refresh = app.update(command.expect_one_event(), &mut model);
        let current = state(&app, &model);
        assert!(current.saving && current.loading);
        if deleting {
            assert!(current.servers.is_empty());
        } else {
            assert!(!current.servers[0].enabled && current.servers[0].tools.is_empty());
        }
        reply(&app, &mut model, refresh, success(EMPTY))
            .expect_one_effect()
            .expect_render();
        assert!(!state(&app, &model).saving);
    }
}

#[test]
fn mismatched_write_confirmation_blocks_mutations_until_manual_get() {
    let app = AppCore::default();
    for (action, ack) in [(create(), DELETED), (toggle(false), ACK), (delete(), ACK)] {
        let mut model = loaded(&app);
        draft(&app, &mut model);
        let before = state(&app, &model).servers;
        let command = app.update(action, &mut model);
        reply(&app, &mut model, command, success(ack))
            .expect_one_effect()
            .expect_render();
        let current = state(&app, &model);
        assert!(current.write_uncertain && !current.saving && current.draft_present);
        assert_eq!(current.servers, before);
        assert!(current.error.as_deref().unwrap().contains("不一致"));
        for action in [create(), toggle(false), delete()] {
            assert!(app.update(action, &mut model).is_done());
        }
        let command = app.update(get(), &mut model);
        reply(&app, &mut model, command, success(LIST))
            .expect_one_effect()
            .expect_render();
        assert!(!state(&app, &model).write_uncertain);
        assert!(
            state(&app, &model)
                .notice
                .as_deref()
                .unwrap()
                .contains("核对上次操作")
        );
    }
}

#[test]
fn timeout_network_json_and_5xx_preserve_private_draft_and_require_refresh() {
    let app = AppCore::default();
    for failure in [
        HttpResult::Err(HttpError::Timeout),
        HttpResult::Err(HttpError::Io("TEST_SECRET".into())),
        success("null"),
        success("invalid TEST_SECRET"),
        HttpResult::Ok(HttpResponse::status(500).body("TEST_SECRET").build()),
    ] {
        let mut model = loaded(&app);
        draft(&app, &mut model);
        let command = app.update(create(), &mut model);
        reply(&app, &mut model, command, failure)
            .expect_one_effect()
            .expect_render();
        let current = state(&app, &model);
        assert!(current.write_uncertain && current.draft_present && !current.saving);
        assert!(!format!("{current:?}").contains("TEST_SECRET"));
        assert!(app.update(create(), &mut model).is_done());
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
        let command = app.update(get(), &mut model);
        reply(&app, &mut model, command, success(LIST))
            .expect_one_effect()
            .expect_render();
        let mut retry = app.update(create(), &mut model);
        retry.expect_effect().expect_render();
        let request = retry.expect_one_effect().expect_http();
        assert_eq!(
            serde_json::from_slice::<Value>(&request.operation.body).unwrap(),
            serde_json::from_str::<Value>(CONFIG).unwrap()
        );
    }
}

#[test]
fn known_404_and_422_errors_allow_corrected_manual_retry() {
    let app = AppCore::default();
    for code in [404, 422] {
        let mut model = loaded(&app);
        let command = app.update(toggle(false), &mut model);
        reply(
            &app,
            &mut model,
            command,
            HttpResult::Ok(HttpResponse::status(code).body("TEST_SECRET").build()),
        )
        .expect_one_effect()
        .expect_render();
        let current = state(&app, &model);
        assert!(!current.write_uncertain && !current.saving);
        assert!(current.servers[0].enabled);
        assert!(!format!("{current:?}").contains("TEST_SECRET"));
        if code == 404 {
            assert!(current.error.unwrap().contains("刷新列表"));
        }
        assert!(!app.update(toggle(false), &mut model).is_done());
    }
}

#[test]
fn busy_guards_cover_post_and_post_refresh_and_reset_is_allowed_during_read() {
    let app = AppCore::default();
    for phase in 0..3 {
        let mut model = loaded(&app);
        draft(&app, &mut model);
        let command = if phase == 0 {
            app.update(get(), &mut model)
        } else {
            app.update(create(), &mut model)
        };
        let _in_flight = if phase == 2 {
            reply(&app, &mut model, command, success(ACK))
        } else {
            command
        };
        let before = state(&app, &model);
        for action in [get(), create(), toggle(false), delete(), edit("{}")] {
            assert!(app.update(action, &mut model).is_done());
            assert_eq!(state(&app, &model), before);
        }
        let mut reset = app.update(event(McpConfigEvent::ResetDraft), &mut model);
        if phase == 0 {
            reset.expect_one_effect().expect_render();
            assert!(!state(&app, &model).draft_present);
        } else {
            assert!(reset.is_done());
            assert_eq!(state(&app, &model), before);
        }
    }
}

#[test]
fn reset_clears_private_input_without_erasing_uncertain_write_guard() {
    let app = AppCore::default();
    let mut model = loaded(&app);
    draft(&app, &mut model);
    let command = app.update(create(), &mut model);
    reply(
        &app,
        &mut model,
        command,
        HttpResult::Err(HttpError::Timeout),
    )
    .expect_one_effect()
    .expect_render();
    app.update(event(McpConfigEvent::ResetDraft), &mut model)
        .expect_one_effect()
        .expect_render();
    let current = state(&app, &model);
    assert!(!current.draft_present && current.error.is_none() && current.write_uncertain);
    let command = app.update(get(), &mut model);
    reply(&app, &mut model, command, success(LIST))
        .expect_one_effect()
        .expect_render();
    app.update(create(), &mut model)
        .expect_one_effect()
        .expect_render();
    assert!(state(&app, &model).error.is_some());
}

#[test]
fn failed_reads_preserve_data_and_draft_and_invalid_service_urls_send_no_request() {
    let app = AppCore::default();
    for existing in [false, true] {
        let mut model = if existing {
            loaded(&app)
        } else {
            Model::default()
        };
        draft(&app, &mut model);
        let before = state(&app, &model);
        let command = app.update(get(), &mut model);
        reply(
            &app,
            &mut model,
            command,
            success(r#"{"mcp_servers":[{}]}"#),
        )
        .expect_one_effect()
        .expect_render();
        let current = state(&app, &model);
        assert_eq!(current.servers, before.servers);
        assert_eq!(current.loaded, existing);
        assert!(current.draft_present && current.error.is_some() && !current.loading);
        let command = app.update(get(), &mut model);
        reply(&app, &mut model, command, success(LIST))
            .expect_one_effect()
            .expect_render();
        app.update(
            event(McpConfigEvent::Create {
                base_url: "file:///tmp".to_string(),
            }),
            &mut model,
        )
        .expect_one_effect()
        .expect_render();
        assert!(!state(&app, &model).saving);
        app.update(
            event(McpConfigEvent::Get {
                base_url: "bad".to_string(),
            }),
            &mut model,
        )
        .expect_one_effect()
        .expect_render();
        assert!(!state(&app, &model).loading);
    }
}

#[test]
fn rejects_missing_names_and_unchanged_operations_without_mutating_state() {
    let app = AppCore::default();
    let mut model = Model::default();
    for action in [create(), toggle(false), delete()] {
        assert!(app.update(action, &mut model).is_done());
    }
    let mut model = loaded(&app);
    assert!(app.update(toggle(true), &mut model).is_done());
    for action in [
        event(McpConfigEvent::Delete {
            base_url: BASE.to_string(),
            server_name: "missing".to_string(),
        }),
        event(McpConfigEvent::SetEnabled {
            base_url: BASE.to_string(),
            server_name: "missing".to_string(),
            enabled: false,
        }),
    ] {
        app.update(action, &mut model)
            .expect_one_effect()
            .expect_render();
        assert!(state(&app, &model).error.unwrap().contains("刷新列表"));
    }
}

#[test]
fn confirmation_keeps_unrelated_server_tools_and_clears_updated_server_tools() {
    let app = AppCore::default();
    let mut model = loaded(&app);
    let config = r#"{"mcpServers":{"new":{"transport":"stdio","command":"node"}}}"#;
    app.update(edit(config), &mut model)
        .expect_one_effect()
        .expect_render();
    let command = app.update(create(), &mut model);
    let ack = r#"{"mcpServers":{"demo":{"transport":"stdio","enabled":true},"new":{"transport":"stdio","enabled":true}}}"#;
    let refresh = reply(&app, &mut model, command, success(ack));
    let current = state(&app, &model);
    assert_eq!(current.servers[0].tools, ["search"]);
    assert!(current.servers[1].tools.is_empty());
    reply(
        &app,
        &mut model,
        refresh,
        HttpResult::Err(HttpError::Timeout),
    )
    .expect_one_effect()
    .expect_render();
    let current = state(&app, &model);
    assert_eq!(current.servers[0].tools, ["search"]);
    assert!(current.servers[1].tools.is_empty());
}
