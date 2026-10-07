pub mod api;
pub mod app;
mod capabilities;
pub mod effects;
#[cfg(feature = "ffi")]
mod ffi;
pub mod model;
pub mod view;

pub use api::configs::{a2a::A2aServer, agent::AgentConfig, llm::LlmConfig, mcp::McpServer};
pub use app::AppCore;
pub use capabilities::sse;
pub use effects::Effect;
pub use model::configs::{
    a2a::A2aConfigEvent,
    agent::{AgentConfigDraft, AgentConfigEvent, AgentConfigField},
    llm::{LlmConfigDraft, LlmConfigEvent, LlmConfigField},
    mcp::McpConfigEvent,
};
pub use model::{ConfigsEvent, Count, Event, Model};
pub use view::{
    A2aConfigViewModel, AgentConfigViewModel, LlmConfigViewModel, McpConfigViewModel, ViewModel,
};

pub use crux_core::Core;
pub use crux_http as http;
pub use crux_kv as key_value;
pub use crux_time as time;

#[cfg(feature = "ffi")]
pub use ffi::CoreFfi;
