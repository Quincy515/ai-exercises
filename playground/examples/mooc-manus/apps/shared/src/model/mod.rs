//! 根事件与状态：分发业务事件，并处理跨模块协调。

pub mod configs;

use std::time::SystemTime;

use chrono::serde::ts_milliseconds_option::deserialize as ts_milliseconds_option;
use chrono::{DateTime, Utc};
use crux_core::{Command, render::render};
use crux_http::{Url, command::Http};
use crux_kv::command::KeyValue;
use crux_time::command::Time;
use facet::Facet;
use serde::{Deserialize, Serialize};

use crate::{effects::Effect, sse};
pub use configs::ConfigsEvent;
use configs::ConfigsModel;

const KEY: &str = "state";
const API_URL: &str = "https://crux-counter.fly.dev";

#[derive(Default, Serialize, Deserialize)]
pub struct Model {
    pub(crate) count: Count,
    time: Option<String>,
    // 请求状态只在本次运行中使用，沿用计数器原有的 KV 数据格式。
    #[serde(skip)]
    pub(crate) configs: ConfigsModel,
}

#[derive(Serialize, Deserialize, Clone, Default, Debug, PartialEq, Eq)]
pub struct Count {
    pub(crate) value: isize,
    #[serde(deserialize_with = "ts_milliseconds_option")]
    pub(crate) updated_at: Option<DateTime<Utc>>,
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
    Configs(ConfigsEvent),

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

impl Model {
    pub fn update(&mut self, event: Event) -> Command<Effect, Event> {
        match event {
            Event::None => Command::done(),
            Event::Configs(event) => self.configs.update(event).map_event(Event::Configs),
            Event::LoadState => KeyValue::get(KEY).then_send(Event::SetState),
            Event::SetState(Ok(Some(value))) => {
                match serde_json::from_slice::<Model>(&value) {
                    Ok(m) => {
                        // KV 只恢复计数器，保留当前 Agent 配置请求状态。
                        self.count = m.count;
                        self.time = m.time;
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
                // Keep the current view usable when the shell reports an HTTP failure.
                tracing::warn!(error = %e, "Error getting count");
                render()
            }
            Event::Update(count) => {
                self.count = count;
                Time::now().then_send(Event::CurrentTime).and(render())
            }
            Event::Increment => {
                // optimistic update
                self.count = Count {
                    value: self.count.value + 1,
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
                self.count = Count {
                    value: self.count.value - 1,
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
                self.count = Count {
                    value: 0,
                    updated_at: None,
                };

                render()
            }
            Event::CurrentTime(time) => {
                let time: DateTime<Utc> = time.into();
                self.time = Some(time.to_rfc3339_opts(chrono::SecondsFormat::Secs, true));

                Command::all([render()])
            }
            Event::StartWatch => {
                let base = Url::parse(API_URL).unwrap();
                let url = base.join("/sse").unwrap();
                sse::get(url).then_send(Event::Update)
            }
        }
    }
}

#[cfg(test)]
mod tests;
