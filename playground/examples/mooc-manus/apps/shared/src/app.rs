use std::time::SystemTime;

use chrono::serde::ts_milliseconds_option::deserialize as ts_milliseconds_option;
use chrono::{DateTime, Utc};
use crux_core::{
    App, Command,
    macros::effect,
    render::{RenderOperation, render},
};
use crux_http::Url;
use crux_http::{command::Http, protocol::HttpRequest};
use crux_kv::{KeyValueOperation, command::KeyValue};
use crux_time::{TimeRequest, command::Time};
use facet::Facet;
use serde::{Deserialize, Serialize};

use crate::sse::{self, SseRequest};

const KEY: &str = "state";
const API_URL: &str = "https://crux-counter.fly.dev";

#[derive(Default, Serialize, Deserialize)]
pub struct Model {
    count: Count,
    time: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Default, Debug, PartialEq, Eq)]
pub struct Count {
    value: isize,
    #[serde(deserialize_with = "ts_milliseconds_option")]
    updated_at: Option<DateTime<Utc>>,
}

#[derive(Facet, Serialize, Deserialize, Debug, Clone)]
pub struct ViewModel {
    pub text: String,
    pub confirmed: bool,
}

#[derive(Facet, Serialize, Deserialize, Debug, PartialEq, Eq)]
#[repr(C)]
pub enum Event {
    None,
    Get,
    Increment,
    Decrement,
    Reset,
    LoadState,
    StartWatch,

    // events local to the core:
    // we can skip serialization using `#[serde(skip)]`
    // we can skip typegen using `#[facet(skip)]`
    // we can use types that do not implement `Facet`, by marking them with `#[facet(opaque)]`
    #[serde(skip)]
    #[facet(skip)]
    CurrentTime(#[facet(opaque)] SystemTime),
    #[serde(skip)]
    #[facet(skip)]
    SetState(#[facet(opaque)] crux_kv::DataResult),
    #[serde(skip)]
    #[facet(skip)]
    Update(#[facet(opaque)] Count),
    #[serde(skip)]
    #[facet(skip)]
    Set(#[facet(opaque)] crux_http::Result<crux_http::Response<Count>>),
}

#[effect(facet_typegen)]
#[derive(Debug)]
pub enum Effect {
    Render(RenderOperation),
    Http(HttpRequest),
    Time(TimeRequest),
    KeyValue(KeyValueOperation),
    ServerSentEvents(SseRequest),
}

#[derive(Default)]
pub struct AppCore {}

impl App for AppCore {
    type Event = Event;
    type Model = Model;
    type ViewModel = ViewModel;
    type Effect = Effect;

    fn update(&self, event: Event, model: &mut Model) -> Command<Effect, Event> {
        match event {
            Event::None => Command::done(),
            Event::LoadState => KeyValue::get(KEY).then_send(Event::SetState),
            Event::SetState(Ok(Some(value))) => {
                match serde_json::from_slice::<Model>(&value) {
                    Ok(m) => {
                        *model = m;
                        render()
                    }
                    Err(_) => {
                        // handle error
                        Command::done()
                    }
                }
            }
            // KV store has no saved state (first launch)
            // KV 存储中无已保存状态（首次启动）
            Event::SetState(Ok(None)) => render(),
            // KV read failed, continue with default state
            // KV 读取失败，使用默认状态继续
            Event::SetState(Err(_)) => render(),
            Event::Get => Http::get(API_URL)
                .expect_json()
                .build()
                .then_send(Event::Set),
            Event::Set(Ok(mut response)) => {
                let count = response.take_body().unwrap();
                Command::event(Event::Update(count))
            }
            Event::Set(Err(e)) => {
                panic!("Error getting count: {e}");
            }
            Event::Update(count) => {
                model.count = count;
                Time::now().then_send(Event::CurrentTime).and(render())
            }
            Event::Increment => {
                // optimistic update
                model.count = Count {
                    value: model.count.value + 1,
                    updated_at: None,
                };

                let call_api = {
                    let base = Url::parse(API_URL).unwrap();
                    let url = base.join("/inc").unwrap();
                    Http::post(url).expect_json().build().then_send(Event::Set)
                };

                render().and(call_api)
            }
            Event::Decrement => {
                // optimistic update
                model.count = Count {
                    value: model.count.value - 1,
                    updated_at: None,
                };

                let call_api = {
                    let base = Url::parse(API_URL).unwrap();
                    let url = base.join("/dec").unwrap();
                    Http::post(url).expect_json().build().then_send(Event::Set)
                };

                render().and(call_api)
            }
            Event::Reset => {
                // optimistic update
                model.count = Count {
                    value: 0,
                    updated_at: None,
                };

                render()
            }
            Event::CurrentTime(time) => {
                let time: DateTime<Utc> = time.into();
                model.time = Some(time.to_rfc3339_opts(chrono::SecondsFormat::Secs, true));

                Command::all([render()])
            }
            Event::StartWatch => {
                let base = Url::parse(API_URL).unwrap();
                let url = base.join("/sse").unwrap();
                sse::get(url).then_send(Event::Update)
            }
        }
    }

    fn view(&self, model: &Self::Model) -> Self::ViewModel {
        let suffix = model.count.updated_at.map_or_else(
            || " (pending)".to_string(),
            |updated_at| format!(" ({updated_at})"),
        );

        ViewModel {
            text: model.count.value.to_string() + &suffix,
            confirmed: model.count.updated_at.is_some(),
        }
    }
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone, Utc};

    use crux_core::App as _;
    use crux_http::{
        protocol::{HttpRequest, HttpResponse, HttpResult},
        testing::ResponseBuilder,
    };
    use crux_time::{Instant, TimeRequest, TimeResponse};

    use super::{AppCore, Event, Model};
    use crate::Count;

    // ANCHOR: simple_tests
    /// Test that a `Get` event causes the app to fetch the current
    /// counter-value from the web API
    #[test]
    fn get_counter() {
        let app = AppCore::default();
        let mut model = Model::default();

        // send a `Get` event to the app
        let mut cmd = app.update(Event::Get, &mut model);

        // the app should emit an HTTP request to fetch the counter
        let (operation, mut request) = cmd.expect_one_effect().expect_http().split();

        // and the request should be a GET to the correct URL
        assert_eq!(
            operation,
            HttpRequest::get("https://crux-counter.fly.dev/").build()
        );

        // resolve the request with a simulated response from the web API
        request
            .resolve(HttpResult::Ok(
                HttpResponse::ok()
                    .body(r#"{ "value": 1, "updated_at": 1672531200000 }"#)
                    .build(),
            ))
            .unwrap();

        // the app should emit a `Set` event with the HTTP response
        let actual = cmd.expect_one_event();
        let expected = Event::Set(Ok(ResponseBuilder::ok()
            .body(Count {
                value: 1,
                updated_at: Some(Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap()),
            })
            .build()));
        assert_eq!(actual, expected);

        // send the `Set` event back to the app
        let mut cmd = app.update(actual, &mut model);

        // check in flight that the app has not been updated with the server data
        let view = app.view(&model);
        assert_eq!(view.text, "0 (pending)");

        // this should generate an `Update` event
        let event = cmd.expect_one_event();
        assert_eq!(
            event,
            Event::Update(Count {
                value: 1,
                updated_at: Some(Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap()),
            })
        );

        // send the `Update` event back to the app
        let mut cmd = app.update(event, &mut model);

        // the model should be updated
        assert_eq!(
            model.count,
            Count {
                value: 1,
                updated_at: Some(Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap()),
            }
        );

        // the app should ask the shell for the current time
        let mut time_request = cmd.expect_effect().expect_time();
        assert_eq!(time_request.operation, TimeRequest::Now);

        // the app should ask the shell to render
        cmd.expect_one_effect().expect_render();

        // resolve the clock request and update the model with a fixed time
        time_request
            .resolve(TimeResponse::Now {
                instant: Instant::new(1_672_531_200, 0),
            })
            .unwrap();
        let event = cmd.expect_one_event();
        assert!(matches!(event, Event::CurrentTime(_)));
        let mut cmd = app.update(event, &mut model);
        cmd.expect_one_effect().expect_render();
        assert_eq!(model.time.as_deref(), Some("2023-01-01T00:00:00Z"));

        // the view should be updated
        let view = app.view(&model);
        assert_eq!(view.text, "1 (2023-01-01 00:00:00 UTC)");
        assert!(view.confirmed);
    }
    // ANCHOR_END: simple_tests

    // Test that an `Increment` event causes the app to increment the counter
    #[test]
    fn increment_counter() {
        let app = AppCore::default();

        // set up our initial model as though we've previously fetched the counter
        let mut model = Model {
            count: Count {
                value: 1,
                updated_at: Some(Utc.with_ymd_and_hms(2022, 12, 31, 23, 59, 0).unwrap()),
            },
            ..Default::default()
        };

        // send an `Increment` event to the app
        let mut cmd = app.update(Event::Increment, &mut model);

        // the app should ask the shell to render the optimistic update
        cmd.expect_effect().expect_render();

        // and send an HTTP post
        let mut request = cmd.expect_one_effect().expect_http();
        assert_eq!(
            &request.operation,
            &HttpRequest::post("https://crux-counter.fly.dev/inc").build()
        );

        // we are expecting our model to be updated "optimistically" before the
        // HTTP request completes, so the value should have been updated
        // but not the timestamp
        assert_eq!(
            model.count,
            Count {
                value: 2,
                updated_at: None
            }
        );

        // resolve the request with a simulated response from the web API
        request
            .resolve(HttpResult::Ok(
                HttpResponse::ok()
                    .body(r#"{ "value": 2, "updated_at": 1672531200000 }"#)
                    .build(),
            ))
            .unwrap();

        // this should generate a `Set` event
        let event = cmd.expect_one_event();
        assert!(matches!(event, Event::Set(_)));

        // send the `Set` event back to the app
        let mut cmd = app.update(event, &mut model);

        // this should generate an `Update` event
        let event = cmd.expect_one_event();
        assert_eq!(
            event,
            Event::Update(Count {
                value: 2,
                updated_at: Some(Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap()),
            })
        );

        // send the `Update` event back to the app
        let mut cmd = app.update(event, &mut model);

        // the app should ask the shell for the current time
        let time_request = cmd.expect_effect().expect_time();
        assert_eq!(time_request.operation, TimeRequest::Now);

        // the app should ask the shell to render
        cmd.expect_one_effect().expect_render();

        // the model should be updated
        insta::assert_yaml_snapshot!(model, @r#"
        count:
          value: 2
          updated_at: "2023-01-01T00:00:00Z"
        time: ~
        "#);
    }

    /// Test that a `Decrement` event causes the app to decrement the counter
    #[test]
    fn decrement_counter() {
        let app = AppCore::default();

        // set up our initial model as though we've previously fetched the counter
        let mut model = Model {
            count: Count {
                value: 0,
                updated_at: Some(Utc.with_ymd_and_hms(2022, 12, 31, 23, 59, 0).unwrap()),
            },
            ..Default::default()
        };

        // send a `Decrement` event to the app
        let mut update = app.update(Event::Decrement, &mut model);

        // the app should ask the shell to render the optimistic update
        update.expect_effect().expect_render();

        // and send an HTTP post
        let mut request = update.expect_one_effect().expect_http();
        assert_eq!(
            &request.operation,
            &HttpRequest::post("https://crux-counter.fly.dev/dec").build()
        );

        // we are expecting our model to be updated "optimistically" before the
        // HTTP request completes, so the value should have been updated
        // but not the timestamp
        assert_eq!(
            model.count,
            Count {
                value: -1,
                updated_at: None
            }
        );

        // resolve the request with a simulated response from the web API
        request
            .resolve(HttpResult::Ok(
                HttpResponse::ok()
                    .body(r#"{ "value": -1, "updated_at": 1672531200000 }"#)
                    .build(),
            ))
            .unwrap();

        // this should generate a `Set` event
        let event = update.expect_one_event();
        assert!(matches!(event, Event::Set(_)));

        // send the `Set` event back to the app
        let mut update = app.update(event, &mut model);

        // this should generate an `Update` event
        let event = update.expect_one_event();
        assert_eq!(
            event,
            Event::Update(Count {
                value: -1,
                updated_at: Some(Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap()),
            })
        );

        // send the `Update` event back to the app
        let mut update = app.update(event, &mut model);

        // the app should ask the shell for the current time
        let time_request = update.expect_effect().expect_time();
        assert_eq!(time_request.operation, TimeRequest::Now);

        // the app should ask the shell to render
        update.expect_one_effect().expect_render();

        // the model should be updated
        insta::assert_yaml_snapshot!(model, @r#"
        count:
          value: -1
          updated_at: "2023-01-01T00:00:00Z"
        time: ~
        "#);
    }
}
