//! 设置业务：管理编辑草稿、输入校验和读写请求的状态转换。

use crux_core::{Command, render::render};
use crux_http::Response;
use facet::Facet;
use serde::{Deserialize, Serialize};

use crate::{
    api::configs::agent::{AgentConfig, get_agent_config, update_agent_config},
    effects::Effect,
};

#[derive(Facet, Serialize, Deserialize, Debug, PartialEq, Eq)]
#[repr(C)]
pub enum AgentConfigEvent {
    Get {
        base_url: String,
    },
    Edit {
        field: AgentConfigField,
        value: String,
    },
    Reset,
    Save {
        base_url: String,
    },

    // 请求结果仅在 Core 内流转。
    #[serde(skip)]
    #[facet(skip)]
    Received(#[facet(opaque)] crux_http::Result<Response<AgentConfig>>),
    #[serde(skip)]
    #[facet(skip)]
    Saved(#[facet(opaque)] crux_http::Result<Response<AgentConfig>>),
}

#[derive(Facet, Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub enum AgentConfigField {
    MaxIterations,
    MaxRetries,
    MaxSearchResults,
}

/// 字符串草稿保留清空、未完成输入等编辑状态，保存时统一校验。
#[derive(Facet, Serialize, Deserialize, Debug, Clone, Default, PartialEq, Eq)]
pub struct AgentConfigDraft {
    pub max_iterations: String,
    pub max_retries: String,
    pub max_search_results: String,
}

impl From<&AgentConfig> for AgentConfigDraft {
    fn from(config: &AgentConfig) -> Self {
        Self {
            max_iterations: config.max_iterations.to_string(),
            max_retries: config.max_retries.to_string(),
            max_search_results: config.max_search_results.to_string(),
        }
    }
}

impl AgentConfigDraft {
    fn validate(&self) -> Result<AgentConfig, String> {
        // 与 server/src/domain/models/app_config.rs 的整数范围保持一致。
        Ok(AgentConfig {
            max_iterations: parse_field(&self.max_iterations, "最大迭代次数", 1, 999)?,
            max_retries: parse_field(&self.max_retries, "最大重试次数", 2, 9)?,
            max_search_results: parse_field(&self.max_search_results, "最大搜索结果数", 2, 29)?,
        })
    }
}

fn parse_field(value: &str, label: &str, min: i64, max: i64) -> Result<i64, String> {
    value
        .trim()
        .parse::<i64>()
        .ok()
        .filter(|value| (min..=max).contains(value))
        .ok_or_else(|| format!("{label}必须是 {min}–{max} 的整数。"))
}

/// 设置模块的运行状态，由 ViewModel 转换为页面数据。
#[derive(Default)]
pub struct AgentConfigModel {
    pub(crate) data: Option<AgentConfig>,
    pub(crate) draft: AgentConfigDraft,
    pub(crate) loading: bool,
    pub(crate) saving: bool,
    pub(crate) error: Option<String>,
    pub(crate) saved: bool,
}

impl AgentConfigModel {
    pub fn update(&mut self, event: AgentConfigEvent) -> Command<Effect, AgentConfigEvent> {
        match event {
            AgentConfigEvent::Get { base_url } => self.fetch(&base_url),
            AgentConfigEvent::Edit { field, value } => self.edit(field, value),
            AgentConfigEvent::Reset => self.reset(),
            AgentConfigEvent::Save { base_url } => self.save(&base_url),
            AgentConfigEvent::Received(result) => {
                self.loading = false;
                self.receive(result, false)
            }
            AgentConfigEvent::Saved(result) => {
                self.saving = false;
                self.receive(result, true)
            }
        }
    }

    pub(crate) fn dirty(&self) -> bool {
        self.data
            .as_ref()
            .is_some_and(|config| self.draft != AgentConfigDraft::from(config))
    }

    pub(crate) fn can_save(&self) -> bool {
        self.dirty() && !self.loading && !self.saving
    }

    fn fetch(&mut self, base_url: &str) -> Command<Effect, AgentConfigEvent> {
        // 串行读写，避免刷新覆盖尚未保存的草稿。
        if self.loading || self.saving || self.dirty() {
            return Command::done();
        }

        let request = match get_agent_config(base_url) {
            Ok(request) => request,
            Err(error) => {
                self.error = Some(error);
                return render();
            }
        };

        self.loading = true;
        self.error = None;
        self.saved = false;

        render().and(request.build().then_send(AgentConfigEvent::Received))
    }

    fn edit(
        &mut self,
        field: AgentConfigField,
        value: String,
    ) -> Command<Effect, AgentConfigEvent> {
        if self.data.is_none() || self.loading || self.saving {
            return Command::done();
        }
        match field {
            AgentConfigField::MaxIterations => self.draft.max_iterations = value,
            AgentConfigField::MaxRetries => self.draft.max_retries = value,
            AgentConfigField::MaxSearchResults => self.draft.max_search_results = value,
        }
        self.error = None;
        self.saved = false;
        render()
    }

    fn reset(&mut self) -> Command<Effect, AgentConfigEvent> {
        if self.loading || self.saving {
            return Command::done();
        }
        self.draft = self
            .data
            .as_ref()
            .map(AgentConfigDraft::from)
            .unwrap_or_default();
        self.error = None;
        self.saved = false;
        render()
    }

    fn save(&mut self, base_url: &str) -> Command<Effect, AgentConfigEvent> {
        if !self.can_save() {
            return Command::done();
        }
        let request = match self
            .draft
            .validate()
            .and_then(|config| update_agent_config(base_url, &config))
        {
            Ok(request) => request,
            Err(error) => {
                self.error = Some(error);
                self.saved = false;
                return render();
            }
        };

        self.saving = true;
        self.error = None;
        self.saved = false;

        render().and(request.build().then_send(AgentConfigEvent::Saved))
    }

    fn receive(
        &mut self,
        result: crux_http::Result<Response<AgentConfig>>,
        saved: bool,
    ) -> Command<Effect, AgentConfigEvent> {
        let config = super::receive_config(result, saved, "Agent");

        match config {
            Ok(config) => {
                self.draft = AgentConfigDraft::from(&config);
                self.data = Some(config);
                self.error = None;
                self.saved = saved;
            }
            // 失败时保留已确认数据与编辑草稿，写请求由用户手动重试。
            Err(error) => self.error = Some(error),
        }

        render()
    }
}

#[cfg(test)]
mod tests {
    use crux_core::{App as _, Command};
    use crux_http::{
        HttpError,
        protocol::{HttpRequest, HttpResponse, HttpResult},
        testing::ResponseBuilder,
    };

    use crate::{AppCore, Effect, Event, Model};

    use super::{AgentConfig, AgentConfigDraft, AgentConfigEvent, AgentConfigField};

    const BASE_URL: &str = "http://localhost:5150";
    const JSON: &str = r#"{"max_iterations":20,"max_retries":3,"max_search_results":10}"#;

    fn get() -> Event {
        Event::Configs(crate::ConfigsEvent::Agent(AgentConfigEvent::Get {
            base_url: BASE_URL.to_string(),
        }))
    }

    fn finish_request(
        app: &AppCore,
        model: &mut Model,
        mut command: Command<Effect, Event>,
        result: HttpResult,
    ) {
        command.expect_effect().expect_render();
        let mut request = command.expect_one_effect().expect_http();
        request.resolve(result).unwrap();
        let event = command.expect_one_event();
        assert!(matches!(
            event,
            Event::Configs(crate::ConfigsEvent::Agent(
                AgentConfigEvent::Received(_) | AgentConfigEvent::Saved(_)
            ))
        ));
        let mut command = app.update(event, model);
        command.expect_one_effect().expect_render();
    }

    #[test]
    fn fetches_exact_endpoint_and_exposes_loading_then_data() {
        let app = AppCore::default();
        let mut model = Model::default();
        let mut command = app.update(get(), &mut model);

        insta::assert_yaml_snapshot!(app.view(&model).agent_config, @r#"
        data: ~
        draft:
          max_iterations: ""
          max_retries: ""
          max_search_results: ""
        loading: true
        saving: false
        error: ~
        saved: false
        dirty: false
        can_save: false
        "#);

        command.expect_effect().expect_render();
        let mut request = command.expect_one_effect().expect_http();
        assert_eq!(
            request.operation,
            HttpRequest::get("http://localhost:5150/api/app_configs/agent").build()
        );
        request
            .resolve(HttpResult::Ok(HttpResponse::ok().body(JSON).build()))
            .unwrap();

        let mut command = app.update(command.expect_one_event(), &mut model);
        command.expect_one_effect().expect_render();
        insta::assert_yaml_snapshot!(app.view(&model).agent_config, @r#"
        data:
          max_iterations: 20
          max_retries: 3
          max_search_results: 10
        draft:
          max_iterations: "20"
          max_retries: "3"
          max_search_results: "10"
        loading: false
        saving: false
        error: ~
        saved: false
        dirty: false
        can_save: false
        "#);
        assert_eq!(app.view(&model).text, "0 (pending)");
    }

    #[test]
    fn ignores_duplicate_fetches_while_loading() {
        let app = AppCore::default();
        let mut model = Model::default();
        let first = app.update(get(), &mut model);
        let mut duplicate = app.update(get(), &mut model);
        assert!(duplicate.is_done());
        assert!(app.view(&model).agent_config.loading);

        finish_request(
            &app,
            &mut model,
            first,
            HttpResult::Ok(HttpResponse::ok().body(JSON).build()),
        );
        assert!(app.view(&model).agent_config.data.is_some());
    }

    #[test]
    fn restoring_counter_state_preserves_the_in_flight_request() {
        let app = AppCore::default();
        let mut model = Model::default();
        let command = app.update(get(), &mut model);
        let saved_counter = br#"{"count":{"value":7,"updated_at":null},"time":null}"#.to_vec();
        let mut restore = app.update(Event::SetState(Ok(Some(saved_counter))), &mut model);
        restore.expect_one_effect().expect_render();
        assert!(app.view(&model).agent_config.loading);
        assert_eq!(app.view(&model).text, "7 (pending)");
        assert_eq!(
            serde_json::to_value(&model).unwrap(),
            serde_json::json!({"count": {"value": 7, "updated_at": null}, "time": null})
        );

        finish_request(
            &app,
            &mut model,
            command,
            HttpResult::Ok(HttpResponse::ok().body(JSON).build()),
        );
        assert!(app.view(&model).agent_config.data.is_some());
    }

    #[test]
    fn converts_transport_http_and_json_failures_to_view_errors() {
        let failures = [
            (
                HttpResult::Err(HttpError::Io("offline".to_string())),
                "网络请求失败，请检查后端服务和网络连接。",
            ),
            (HttpResult::Err(HttpError::Timeout), "请求超时，请重试。"),
            (
                HttpResult::Ok(HttpResponse::status(500).body("internal error").build()),
                "读取 Agent 配置失败（HTTP 500），请重试。",
            ),
            (
                HttpResult::Ok(HttpResponse::ok().body("invalid JSON").build()),
                "配置响应格式有误，请检查接口字段。",
            ),
            (
                HttpResult::Ok(HttpResponse::ok().body(r#"{"max_retries":3}"#).build()),
                "配置响应格式有误，请检查接口字段。",
            ),
            (
                HttpResult::Ok(HttpResponse::ok().build()),
                "配置响应格式有误，请检查接口字段。",
            ),
        ];

        for (result, message) in failures {
            let app = AppCore::default();
            let mut model = Model::default();
            let command = app.update(get(), &mut model);
            finish_request(&app, &mut model, command, result);
            let state = app.view(&model).agent_config;
            assert!(!state.loading);
            assert_eq!(state.data, None);
            assert_eq!(state.error.as_deref(), Some(message));
        }
    }

    #[test]
    fn retains_previous_data_on_failure_and_allows_retry() {
        let app = AppCore::default();
        let mut model = Model::default();
        let command = app.update(get(), &mut model);
        finish_request(
            &app,
            &mut model,
            command,
            HttpResult::Ok(HttpResponse::ok().body(JSON).build()),
        );
        let previous = app.view(&model).agent_config.data;

        let command = app.update(get(), &mut model);
        finish_request(
            &app,
            &mut model,
            command,
            HttpResult::Err(HttpError::Io("offline".to_string())),
        );
        let state = app.view(&model).agent_config;
        assert_eq!(state.data, previous);
        assert!(state.error.is_some());

        let command = app.update(get(), &mut model);
        let state = app.view(&model).agent_config;
        assert!(state.loading);
        assert_eq!(state.error, None);
        assert_eq!(state.data, previous);
        finish_request(
            &app,
            &mut model,
            command,
            HttpResult::Ok(
                HttpResponse::ok()
                    .body(r#"{"max_iterations":30,"max_retries":4,"max_search_results":12}"#)
                    .build(),
            ),
        );
        let state = app.view(&model).agent_config;
        assert_eq!(state.data.unwrap().max_iterations, 30);
        assert!(!state.loading);
        assert_eq!(state.error, None);
    }

    #[test]
    fn handles_a_response_whose_body_has_already_been_taken() {
        let app = AppCore::default();
        let mut model = Model::default();
        let mut response = ResponseBuilder::ok()
            .body(AgentConfig {
                max_iterations: 20,
                max_retries: 3,
                max_search_results: 10,
            })
            .build();
        response.take_body();

        let mut command = app.update(
            Event::Configs(crate::ConfigsEvent::Agent(AgentConfigEvent::Received(Ok(
                response,
            )))),
            &mut model,
        );
        command.expect_one_effect().expect_render();
        assert_eq!(
            app.view(&model).agent_config.error.as_deref(),
            Some("服务返回了空响应，请重试。")
        );
    }

    #[test]
    fn rejects_invalid_base_url_without_sending_http() {
        let app = AppCore::default();
        for base_url in ["", "invalid", "file:///tmp/settings"] {
            let mut model = Model::default();
            let mut command = app.update(
                Event::Configs(crate::ConfigsEvent::Agent(AgentConfigEvent::Get {
                    base_url: base_url.to_string(),
                })),
                &mut model,
            );
            command.expect_one_effect().expect_render();
            let state = app.view(&model).agent_config;
            assert!(!state.loading);
            assert!(state.error.is_some());
        }
    }

    fn loaded_model(app: &AppCore) -> Model {
        let mut model = Model::default();
        let command = app.update(get(), &mut model);
        finish_request(
            app,
            &mut model,
            command,
            HttpResult::Ok(HttpResponse::ok().body(JSON).build()),
        );
        model
    }

    fn edit(field: AgentConfigField, value: &str) -> Event {
        Event::Configs(crate::ConfigsEvent::Agent(AgentConfigEvent::Edit {
            field,
            value: value.to_string(),
        }))
    }

    fn save() -> Event {
        Event::Configs(crate::ConfigsEvent::Agent(AgentConfigEvent::Save {
            base_url: BASE_URL.to_string(),
        }))
    }

    #[test]
    fn saves_json_and_uses_the_server_response_as_confirmed_data() {
        let app = AppCore::default();
        let mut model = loaded_model(&app);
        app.update(edit(AgentConfigField::MaxIterations, "25"), &mut model)
            .expect_one_effect()
            .expect_render();
        let edited = app.view(&model).agent_config;
        assert_eq!(edited.data.unwrap().max_iterations, 20);
        assert!(edited.dirty && edited.can_save);

        let mut command = app.update(save(), &mut model);
        let saving = app.view(&model).agent_config;
        assert!(saving.saving && saving.dirty);
        assert!(!saving.loading && !saving.can_save && !saving.saved);
        command.expect_effect().expect_render();
        let mut request = command.expect_one_effect().expect_http();
        assert_eq!(
            request.operation,
            HttpRequest::post("http://localhost:5150/api/app_configs/agent")
                .body_json(AgentConfig {
                    max_iterations: 25,
                    max_retries: 3,
                    max_search_results: 10,
                })
                .build()
        );
        request
            .resolve(HttpResult::Ok(
                HttpResponse::ok()
                    .body(r#"{"max_iterations":24,"max_retries":3,"max_search_results":10}"#)
                    .build(),
            ))
            .unwrap();
        app.update(command.expect_one_event(), &mut model)
            .expect_one_effect()
            .expect_render();
        let state = app.view(&model).agent_config;
        assert_eq!(state.data.unwrap().max_iterations, 24);
        assert_eq!(state.draft.max_iterations, "24");
        assert!(state.saved);
        assert!(!state.saving && !state.dirty && !state.can_save);
        assert_eq!(state.error, None);
        assert_eq!(app.view(&model).text, "0 (pending)");
    }

    #[test]
    fn accepts_inclusive_server_validation_boundaries() {
        let app = AppCore::default();
        for (field, min, max) in [
            (AgentConfigField::MaxIterations, 1, 999),
            (AgentConfigField::MaxRetries, 2, 9),
            (AgentConfigField::MaxSearchResults, 2, 29),
        ] {
            for value in [min, max] {
                let mut model = loaded_model(&app);
                app.update(edit(field, &value.to_string()), &mut model)
                    .expect_one_effect()
                    .expect_render();
                let mut command = app.update(save(), &mut model);
                command.expect_effect().expect_render();
                let request = command.expect_one_effect().expect_http();
                let body: AgentConfig = serde_json::from_slice(&request.operation.body).unwrap();
                let actual = match field {
                    AgentConfigField::MaxIterations => body.max_iterations,
                    AgentConfigField::MaxRetries => body.max_retries,
                    AgentConfigField::MaxSearchResults => body.max_search_results,
                };
                assert_eq!(actual, value);
            }
        }
    }

    #[test]
    fn rejects_invalid_and_out_of_range_drafts_without_http() {
        let app = AppCore::default();
        for (field, label, min, max) in [
            (AgentConfigField::MaxIterations, "最大迭代次数", 1, 999),
            (AgentConfigField::MaxRetries, "最大重试次数", 2, 9),
            (AgentConfigField::MaxSearchResults, "最大搜索结果数", 2, 29),
        ] {
            let mut invalid: Vec<String> = [
                "",
                " ",
                "abc",
                "1.5",
                "1e2",
                "NaN",
                "-1",
                "9223372036854775808",
                "-9223372036854775809",
            ]
            .into_iter()
            .map(str::to_string)
            .collect();
            invalid.extend([(min - 1).to_string(), (max + 1).to_string()]);
            for value in invalid {
                let mut model = loaded_model(&app);
                let previous = app.view(&model).agent_config.data;
                app.update(edit(field, &value), &mut model)
                    .expect_one_effect()
                    .expect_render();
                let draft = app.view(&model).agent_config.draft;
                app.update(save(), &mut model)
                    .expect_one_effect()
                    .expect_render();
                let state = app.view(&model).agent_config;
                assert_eq!(
                    state.error,
                    Some(format!("{label}必须是 {min}–{max} 的整数。"))
                );
                assert_eq!(state.data, previous);
                assert_eq!(state.draft, draft);
                assert!(state.dirty && state.can_save);
                assert!(!state.saving && !state.saved);
            }
        }
    }

    #[test]
    fn retains_draft_on_save_failure_and_supports_manual_retry() {
        let app = AppCore::default();
        for (result, message) in [
            (
                HttpResult::Err(HttpError::Io("offline".to_string())),
                "保存时网络异常，结果尚未确认，请稍后重试或重新打开设置核对。",
            ),
            (
                HttpResult::Err(HttpError::Timeout),
                "保存请求超时，结果尚未确认，请稍后重试或重新打开设置核对。",
            ),
            (
                HttpResult::Ok(HttpResponse::status(422).body("validation error").build()),
                "配置未通过服务端校验，请检查输入范围后重试。",
            ),
            (
                HttpResult::Ok(HttpResponse::status(500).body("internal error").build()),
                "保存 Agent 配置失败（HTTP 500），请重试。",
            ),
            (
                HttpResult::Ok(HttpResponse::ok().body("invalid JSON").build()),
                "配置响应格式有误，请检查接口字段。",
            ),
        ] {
            let mut model = loaded_model(&app);
            let previous = app.view(&model).agent_config.data;
            app.update(edit(AgentConfigField::MaxRetries, "4"), &mut model)
                .expect_one_effect()
                .expect_render();
            let command = app.update(save(), &mut model);
            finish_request(&app, &mut model, command, result);
            let state = app.view(&model).agent_config;
            assert_eq!(state.error.as_deref(), Some(message));
            assert_eq!(state.data, previous);
            assert_eq!(state.draft.max_retries, "4");
            assert!(!state.saving && !state.saved);
            assert!(state.dirty && state.can_save);

            let command = app.update(save(), &mut model);
            assert_eq!(app.view(&model).agent_config.error, None);
            finish_request(
                &app,
                &mut model,
                command,
                HttpResult::Ok(
                    HttpResponse::ok()
                        .body(r#"{"max_iterations":20,"max_retries":4,"max_search_results":10}"#)
                        .build(),
                ),
            );
            let state = app.view(&model).agent_config;
            assert!(state.saved);
            assert!(!state.saving && !state.dirty);
            assert_eq!(state.error, None);
        }
    }

    #[test]
    fn ignores_edit_and_save_before_load_and_unchanged_save() {
        let app = AppCore::default();
        let mut model = Model::default();
        assert!(app.update(save(), &mut model).is_done());
        assert!(
            app.update(edit(AgentConfigField::MaxIterations, "30"), &mut model)
                .is_done()
        );
        assert_eq!(
            app.view(&model).agent_config.draft,
            AgentConfigDraft::default()
        );

        let mut model = loaded_model(&app);
        assert!(app.update(save(), &mut model).is_done());
        assert!(!app.view(&model).agent_config.can_save);
    }

    #[test]
    fn guards_edits_reset_refresh_and_duplicate_save_while_busy() {
        let app = AppCore::default();
        for saving in [false, true] {
            let mut model = loaded_model(&app);
            let command = if saving {
                app.update(edit(AgentConfigField::MaxIterations, "25"), &mut model)
                    .expect_one_effect()
                    .expect_render();
                app.update(save(), &mut model)
            } else {
                app.update(get(), &mut model)
            };
            let before = app.view(&model).agent_config;
            for event in [
                get(),
                save(),
                edit(AgentConfigField::MaxIterations, "99"),
                Event::Configs(crate::ConfigsEvent::Agent(AgentConfigEvent::Reset)),
            ] {
                assert!(app.update(event, &mut model).is_done());
                assert_eq!(app.view(&model).agent_config, before);
            }
            finish_request(
                &app,
                &mut model,
                command,
                HttpResult::Ok(HttpResponse::ok().body(JSON).build()),
            );
        }
    }

    #[test]
    fn refresh_preserves_unsaved_draft_and_reset_restores_confirmed_values() {
        let app = AppCore::default();
        let mut model = loaded_model(&app);
        app.update(edit(AgentConfigField::MaxSearchResults, ""), &mut model)
            .expect_one_effect()
            .expect_render();
        app.update(save(), &mut model)
            .expect_one_effect()
            .expect_render();
        let before = app.view(&model).agent_config;
        assert!(app.update(get(), &mut model).is_done());
        assert_eq!(app.view(&model).agent_config, before);

        app.update(
            Event::Configs(crate::ConfigsEvent::Agent(AgentConfigEvent::Reset)),
            &mut model,
        )
        .expect_one_effect()
        .expect_render();
        let state = app.view(&model).agent_config;
        assert_eq!(state.draft.max_search_results, "10");
        assert!(!state.dirty && !state.can_save && !state.saved);
        assert_eq!(state.error, None);
        let command = app.update(get(), &mut model);
        finish_request(
            &app,
            &mut model,
            command,
            HttpResult::Ok(HttpResponse::ok().body(JSON).build()),
        );
    }

    #[test]
    fn editing_clears_validation_and_success_feedback() {
        let app = AppCore::default();
        let mut model = loaded_model(&app);
        app.update(edit(AgentConfigField::MaxIterations, ""), &mut model)
            .expect_one_effect()
            .expect_render();
        app.update(save(), &mut model)
            .expect_one_effect()
            .expect_render();
        assert!(app.view(&model).agent_config.error.is_some());
        app.update(edit(AgentConfigField::MaxIterations, "25"), &mut model)
            .expect_one_effect()
            .expect_render();
        assert_eq!(app.view(&model).agent_config.error, None);
        let command = app.update(save(), &mut model);
        finish_request(
            &app,
            &mut model,
            command,
            HttpResult::Ok(HttpResponse::ok().body(JSON).build()),
        );
        assert!(app.view(&model).agent_config.saved);
        app.update(edit(AgentConfigField::MaxRetries, "4"), &mut model)
            .expect_one_effect()
            .expect_render();
        assert!(!app.view(&model).agent_config.saved);
    }

    #[test]
    fn rejects_invalid_save_url_and_preserves_the_draft() {
        let app = AppCore::default();
        let mut model = loaded_model(&app);
        app.update(edit(AgentConfigField::MaxIterations, "25"), &mut model)
            .expect_one_effect()
            .expect_render();
        app.update(
            Event::Configs(crate::ConfigsEvent::Agent(AgentConfigEvent::Save {
                base_url: "file:///tmp/settings".to_string(),
            })),
            &mut model,
        )
        .expect_one_effect()
        .expect_render();
        let state = app.view(&model).agent_config;
        assert!(state.error.is_some());
        assert_eq!(state.draft.max_iterations, "25");
        assert!(!state.saving && state.can_save);
    }
}
