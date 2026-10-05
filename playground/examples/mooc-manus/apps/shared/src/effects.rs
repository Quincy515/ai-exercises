//! Core 请求 Shell 执行的能力协议，执行逻辑由各端 Shell 提供。

use crux_core::{macros::effect, render::RenderOperation};
use crux_http::protocol::HttpRequest;
use crux_kv::KeyValueOperation;
use crux_time::TimeRequest;

use crate::sse::SseRequest;

#[effect(facet_typegen)]
#[derive(Debug)]
pub enum Effect {
    Render(RenderOperation),
    Http(HttpRequest),
    Time(TimeRequest),
    KeyValue(KeyValueOperation),
    ServerSentEvents(SseRequest),
}
