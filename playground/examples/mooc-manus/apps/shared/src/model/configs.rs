//! 设置业务：处理事件、请求去重和成功/失败状态转换。

use crux_core::{Command, render::render};
use crux_http::{HttpError, Response};
use facet::Facet;
use serde::{Deserialize, Serialize};

use crate::{
    api::configs::{AgentConfig, get_agent_config},
    effects::Effect,
};

#[derive(Facet, Serialize, Deserialize, Debug, PartialEq, Eq)]
#[repr(C)]
pub enum ConfigsEvent {
    GetAgentConfig {
        base_url: String,
    },

    // 请求结果仅在 Core 内流转。
    #[serde(skip)]
    #[facet(skip)]
    AgentConfigReceived(#[facet(opaque)] crux_http::Result<Response<AgentConfig>>),
}

/// 设置模块的运行状态，由 ViewModel 转换为页面数据。
#[derive(Default)]
pub struct ConfigsModel {
    pub(crate) data: Option<AgentConfig>,
    pub(crate) loading: bool,
    pub(crate) error: Option<String>,
}

impl ConfigsModel {
    pub fn update(&mut self, event: ConfigsEvent) -> Command<Effect, ConfigsEvent> {
        match event {
            ConfigsEvent::GetAgentConfig { base_url } => self.fetch(&base_url),
            ConfigsEvent::AgentConfigReceived(result) => self.receive(result),
        }
    }

    fn fetch(&mut self, base_url: &str) -> Command<Effect, ConfigsEvent> {
        // 同一 Core 内连续触发时，复用正在执行的请求。
        if self.loading {
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

        render().and(request.build().then_send(ConfigsEvent::AgentConfigReceived))
    }

    fn receive(
        &mut self,
        result: crux_http::Result<Response<AgentConfig>>,
    ) -> Command<Effect, ConfigsEvent> {
        self.loading = false;

        let config = match result {
            Ok(mut response) => response
                .take_body()
                .ok_or_else(|| "服务返回了空响应，请重试。".to_string()),
            Err(HttpError::Http { code, .. }) => {
                Err(format!("读取 Agent 配置失败（HTTP {code}），请重试。"))
            }
            Err(HttpError::Json(_)) => Err("配置响应格式有误，请检查接口字段。".to_string()),
            Err(HttpError::Timeout) => Err("请求超时，请重试。".to_string()),
            Err(HttpError::Io(_)) => Err("网络请求失败，请检查后端服务和网络连接。".to_string()),
            Err(HttpError::Url(_)) => Err("服务地址无效，请检查 API 地址配置。".to_string()),
            Err(_) => Err("读取 Agent 配置失败，请重试。".to_string()),
        };

        match config {
            Ok(config) => {
                self.data = Some(config);
                self.error = None;
            }
            // 刷新失败时保留最近一次成功的数据，错误和重试状态单独展示。
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

    use super::{AgentConfig, ConfigsEvent};

    const BASE_URL: &str = "http://localhost:5150";
    const JSON: &str = r#"{"max_iterations":20,"max_retries":3,"max_search_results":10}"#;

    fn get() -> Event {
        Event::Configs(ConfigsEvent::GetAgentConfig {
            base_url: BASE_URL.to_string(),
        })
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
            Event::Configs(ConfigsEvent::AgentConfigReceived(_))
        ));
        let mut command = app.update(event, model);
        command.expect_one_effect().expect_render();
    }

    #[test]
    fn fetches_exact_endpoint_and_exposes_loading_then_data() {
        let app = AppCore::default();
        let mut model = Model::default();
        let mut command = app.update(get(), &mut model);

        insta::assert_yaml_snapshot!(app.view(&model).agent_config, @r"
        data: ~
        loading: true
        error: ~
        ");

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
        insta::assert_yaml_snapshot!(app.view(&model).agent_config, @r"
        data:
          max_iterations: 20
          max_retries: 3
          max_search_results: 10
        loading: false
        error: ~
        ");
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
            Event::Configs(ConfigsEvent::AgentConfigReceived(Ok(response))),
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
                Event::Configs(ConfigsEvent::GetAgentConfig {
                    base_url: base_url.to_string(),
                }),
                &mut model,
            );
            command.expect_one_effect().expect_render();
            let state = app.view(&model).agent_config;
            assert!(!state.loading);
            assert!(state.error.is_some());
        }
    }
}
