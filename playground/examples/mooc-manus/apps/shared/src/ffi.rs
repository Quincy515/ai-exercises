#![allow(clippy::used_underscore_items)]

use std::sync::Arc;

use crux_core::{
    Core,
    bridge::{BincodeFfiFormat, EffectId},
};

#[cfg(not(target_family = "wasm"))]
use std::sync::Weak;

#[cfg(not(target_family = "wasm"))]
use crux_core::effects::{EffectRouter, Routes, routes::Serialized};

#[cfg(target_family = "wasm")]
use crux_core::middleware::{Bridge, Layer as _};

use crate::AppCore;

/// For the Shell to provide.
///
/// `boltffi`'s binding generator parses the source and does not evaluate
/// `#[cfg]`, so the FFI surface (this trait and the `CoreFfi` methods below)
/// must present a single, cfg-independent signature. The native and wasm paths
/// therefore differ only inside method bodies, not in their signatures.
#[boltffi::export]
pub trait CruxShell: Send + Sync {
    /// Called when any effects resulting from an asynchronous process
    /// need processing by the shell.
    ///
    /// The bytes are a serialized vector of requests.
    fn process_effects(&self, bytes: Vec<u8>);
}

// Native effects use the same effects::Effect wire schema as typegen and WASM.
// They are delivered through process_effects; update and resolve return no bytes.

#[cfg(not(target_family = "wasm"))]
#[derive(Clone)]
struct EffectRoutes {
    serialized: Arc<Serialized<AppCore, Self, BincodeFfiFormat>>,
}

#[cfg(not(target_family = "wasm"))]
impl Routes<AppCore> for EffectRoutes {
    fn new(router: Weak<EffectRouter<AppCore, Self>>) -> Self {
        Self {
            serialized: Arc::new(Serialized::new(router)),
        }
    }
}

/// The main interface used by the shell.
///
/// Native targets deliver effects through an EffectRouter and the shell callback.
/// On wasm, a Bridge returns effects synchronously from update and resolve.
pub struct CoreFfi {
    #[cfg(not(target_family = "wasm"))]
    router: Arc<EffectRouter<AppCore, EffectRoutes>>,
    #[cfg(target_family = "wasm")]
    inner: Bridge<Core<AppCore>, BincodeFfiFormat>,
}

#[boltffi::export]
#[allow(clippy::missing_panics_doc, clippy::needless_pass_by_value)]
impl CoreFfi {
    pub fn new(shell: Arc<dyn CruxShell>) -> Self {
        #[cfg(not(target_family = "wasm"))]
        {
            let router = EffectRouter::new(Core::new(), move |routes: EffectRoutes| {
                let shell = shell.clone();

                move |effect| {
                    let bytes = routes
                        .serialized
                        .serialize(effect)
                        .expect("serialized effect request should encode");

                    shell.process_effects(bytes);
                }
            });

            Self { router }
        }

        #[cfg(target_family = "wasm")]
        {
            let inner = Core::<AppCore>::new().bridge::<BincodeFfiFormat>(move |effect_bytes| {
                match effect_bytes {
                    Ok(effect) => shell.process_effects(effect),
                    Err(e) => panic!("{e}"),
                }
            });

            Self { inner }
        }
    }

    #[must_use]
    pub fn update(&self, data: &[u8]) -> Vec<u8> {
        #[cfg(not(target_family = "wasm"))]
        {
            self.router
                .routes
                .serialized
                .update(data)
                .expect("event should deserialize");

            Vec::new()
        }

        #[cfg(target_family = "wasm")]
        {
            let mut effects = Vec::new();
            match self.inner.update(data, &mut effects) {
                Ok(()) => effects,
                Err(e) => panic!("{e}"),
            }
        }
    }

    #[must_use]
    pub fn resolve(&self, effect_id: u32, data: &[u8]) -> Vec<u8> {
        #[cfg(not(target_family = "wasm"))]
        {
            self.router
                .routes
                .serialized
                .resolve(EffectId(effect_id), data)
                .expect("failed to resolve effect");

            Vec::new()
        }

        #[cfg(target_family = "wasm")]
        {
            let mut effects = Vec::new();
            match self.inner.resolve(EffectId(effect_id), data, &mut effects) {
                Ok(()) => effects,
                Err(e) => panic!("{e}"),
            }
        }
    }

    #[must_use]
    pub fn view(&self) -> Vec<u8> {
        #[cfg(not(target_family = "wasm"))]
        {
            self.router
                .routes
                .serialized
                .view()
                .expect("view model should serialize")
        }

        #[cfg(target_family = "wasm")]
        {
            let mut view_model = Vec::new();
            match self.inner.view(&mut view_model) {
                Ok(()) => view_model,
                Err(e) => panic!("{e}"),
            }
        }
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    use crux_core::bridge::{BincodeFfiFormat, FfiFormat, Request};
    use crux_http::protocol::{HttpResponse, HttpResult};
    use crux_kv::KeyValueOperation;
    use crux_time::{Instant, TimeRequest, TimeResponse};

    use super::{CoreFfi, CruxShell};
    use crate::{
        A2aConfigEvent, AgentConfigEvent, AgentConfigField, ConfigsEvent, Event, LlmConfigEvent,
        LlmConfigField, McpConfigEvent, ViewModel, effects::EffectFfi,
    };

    struct RecordingShell(mpsc::Sender<Vec<u8>>);

    impl CruxShell for RecordingShell {
        fn process_effects(&self, bytes: Vec<u8>) {
            self.0.send(bytes).unwrap();
        }
    }

    fn encode(value: &impl serde::Serialize) -> Vec<u8> {
        let mut bytes = Vec::new();
        BincodeFfiFormat::serialize(&mut bytes, value).unwrap();
        bytes
    }

    fn receive(rx: &mpsc::Receiver<Vec<u8>>) -> Request<EffectFfi> {
        let bytes = rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let mut requests: Vec<Request<EffectFfi>> = BincodeFfiFormat::deserialize(&bytes).unwrap();
        assert_eq!(requests.len(), 1);
        requests.pop().unwrap()
    }

    #[test]
    fn native_effects_use_the_generated_app_wire_schema() {
        let (tx, rx) = mpsc::channel();
        let core = CoreFfi::new(Arc::new(RecordingShell(tx)));

        assert!(core.update(&encode(&Event::LoadState)).is_empty());
        assert!(matches!(
            receive(&rx).effect,
            EffectFfi::KeyValue(KeyValueOperation::Get { key }) if key == "state"
        ));

        assert!(core.update(&encode(&Event::StartWatch)).is_empty());
        assert!(matches!(
            receive(&rx).effect,
            EffectFfi::ServerSentEvents(request) if request.url.ends_with("/sse")
        ));

        assert!(core.update(&encode(&Event::Get)).is_empty());
        let request = receive(&rx);
        assert!(matches!(request.effect, EffectFfi::Http(_)));
        let response = HttpResult::Ok(
            HttpResponse::ok()
                .body(r#"{"value":7,"updated_at":1672531200000}"#)
                .build(),
        );
        assert!(core.resolve(request.id.0, &encode(&response)).is_empty());
        let time = receive(&rx);
        assert!(matches!(time.effect, EffectFfi::Time(TimeRequest::Now)));
        assert!(matches!(receive(&rx).effect, EffectFfi::Render(_)));
        let response = TimeResponse::Now {
            instant: Instant::new(1_672_531_200, 0),
        };
        assert!(core.resolve(time.id.0, &encode(&response)).is_empty());
        assert!(matches!(receive(&rx).effect, EffectFfi::Render(_)));
        let view: ViewModel = BincodeFfiFormat::deserialize(&core.view()).unwrap();
        assert_eq!(view.text, "7 (2023-01-01 00:00:00 UTC)");
        assert!(view.confirmed);
    }

    #[test]
    fn native_configs_event_resolves_through_the_nested_model() {
        let (tx, rx) = mpsc::channel();
        let core = CoreFfi::new(Arc::new(RecordingShell(tx)));
        let event = Event::Configs(ConfigsEvent::Agent(AgentConfigEvent::Get {
            base_url: "http://localhost:5150".to_string(),
        }));

        assert!(core.update(&encode(&event)).is_empty());
        assert!(matches!(receive(&rx).effect, EffectFfi::Render(_)));
        let request = receive(&rx);
        assert!(matches!(request.effect, EffectFfi::Http(_)));
        let view: ViewModel = BincodeFfiFormat::deserialize(&core.view()).unwrap();
        assert!(view.agent_config.loading);

        let response = HttpResult::Ok(
            HttpResponse::ok()
                .body(r#"{"max_iterations":20,"max_retries":3,"max_search_results":10}"#)
                .build(),
        );
        assert!(core.resolve(request.id.0, &encode(&response)).is_empty());
        assert!(matches!(receive(&rx).effect, EffectFfi::Render(_)));
        let view: ViewModel = BincodeFfiFormat::deserialize(&core.view()).unwrap();
        assert!(!view.agent_config.loading);
        assert_eq!(view.agent_config.error, None);
        assert_eq!(view.agent_config.data.unwrap().max_iterations, 20);
        assert_eq!(view.text, "0 (pending)");

        let edit = Event::Configs(ConfigsEvent::Agent(AgentConfigEvent::Edit {
            field: AgentConfigField::MaxIterations,
            value: "30".to_string(),
        }));
        assert!(core.update(&encode(&edit)).is_empty());
        assert!(matches!(receive(&rx).effect, EffectFfi::Render(_)));
        let view: ViewModel = BincodeFfiFormat::deserialize(&core.view()).unwrap();
        assert_eq!(view.agent_config.draft.max_iterations, "30");
        assert!(view.agent_config.dirty && view.agent_config.can_save);

        let save = Event::Configs(ConfigsEvent::Agent(AgentConfigEvent::Save {
            base_url: "http://localhost:5150".to_string(),
        }));
        assert!(core.update(&encode(&save)).is_empty());
        assert!(matches!(receive(&rx).effect, EffectFfi::Render(_)));
        let request = receive(&rx);
        let EffectFfi::Http(ref http) = request.effect else {
            panic!("expected a save HTTP request");
        };
        assert_eq!(http.method, "POST");
        assert_eq!(http.url, "http://localhost:5150/api/app_configs/agent");
        let body: serde_json::Value = serde_json::from_slice(&http.body).unwrap();
        assert_eq!(body["max_iterations"], 30);
        let response = HttpResult::Ok(
            HttpResponse::ok()
                .body(r#"{"max_iterations":30,"max_retries":3,"max_search_results":10}"#)
                .build(),
        );
        assert!(core.resolve(request.id.0, &encode(&response)).is_empty());
        assert!(matches!(receive(&rx).effect, EffectFfi::Render(_)));
        let view: ViewModel = BincodeFfiFormat::deserialize(&core.view()).unwrap();
        assert!(view.agent_config.saved);
        assert!(!view.agent_config.saving && !view.agent_config.dirty);
        assert_eq!(view.agent_config.data.unwrap().max_iterations, 30);
    }

    #[test]
    fn native_llm_roundtrip_keeps_the_key_out_of_view_bytes() {
        let (tx, rx) = mpsc::channel();
        let core = CoreFfi::new(Arc::new(RecordingShell(tx)));
        let event = Event::Configs(ConfigsEvent::Llm(LlmConfigEvent::Get {
            base_url: "http://localhost:5150".to_string(),
        }));
        assert!(core.update(&encode(&event)).is_empty());
        assert!(matches!(receive(&rx).effect, EffectFfi::Render(_)));
        let request = receive(&rx);
        let response = HttpResult::Ok(
            HttpResponse::ok()
                .body(r#"{"base_url":null,"model_name":null,"temperature":null,"max_tokens":null,"api_key_configured":false}"#)
                .build(),
        );
        assert!(core.resolve(request.id.0, &encode(&response)).is_empty());
        assert!(matches!(receive(&rx).effect, EffectFfi::Render(_)));
        let secret = "ffi-test-only-key";
        let edit = Event::Configs(ConfigsEvent::Llm(LlmConfigEvent::Edit {
            field: LlmConfigField::ApiKey,
            value: secret.to_string(),
        }));
        assert!(core.update(&encode(&edit)).is_empty());
        assert!(matches!(receive(&rx).effect, EffectFfi::Render(_)));
        let bytes = core.view();
        assert!(
            !bytes
                .windows(secret.len())
                .any(|part| part == secret.as_bytes())
        );
        let view: ViewModel = BincodeFfiFormat::deserialize(&bytes).unwrap();
        assert!(view.llm_config.api_key_changed && view.llm_config.can_save);
        assert_eq!(view.llm_config.draft, crate::LlmConfigDraft::default());

        let save = Event::Configs(ConfigsEvent::Llm(LlmConfigEvent::Save {
            base_url: "http://localhost:5150".to_string(),
        }));
        assert!(core.update(&encode(&save)).is_empty());
        assert!(matches!(receive(&rx).effect, EffectFfi::Render(_)));
        let request = receive(&rx);
        let EffectFfi::Http(http) = request.effect else {
            panic!("expected an LLM save HTTP request");
        };
        assert_eq!(http.method, "POST");
        let body: serde_json::Value = serde_json::from_slice(&http.body).unwrap();
        assert_eq!(body["api_key"], secret);
        let response = HttpResult::Ok(
            HttpResponse::ok()
                .body(r#"{"api_key_configured":true,"max_tokens":9223372036854775807}"#)
                .build(),
        );
        assert!(core.resolve(request.id.0, &encode(&response)).is_empty());
        assert!(matches!(receive(&rx).effect, EffectFfi::Render(_)));
        let view: ViewModel = BincodeFfiFormat::deserialize(&core.view()).unwrap();
        assert!(view.llm_config.saved && !view.llm_config.api_key_changed);
        assert_eq!(
            view.llm_config.data.unwrap().max_tokens,
            Some(i64::MAX as u64)
        );
    }

    #[test]
    fn native_a2a_create_resolves_null_then_refreshes_the_list() {
        let (tx, rx) = mpsc::channel();
        let core = CoreFfi::new(Arc::new(RecordingShell(tx)));
        let base_url = "http://localhost:5150".to_string();
        let event = Event::Configs(ConfigsEvent::A2a(A2aConfigEvent::Get {
            base_url: base_url.clone(),
        }));
        assert!(core.update(&encode(&event)).is_empty());
        assert!(matches!(receive(&rx).effect, EffectFfi::Render(_)));
        let request = receive(&rx);
        let empty = HttpResult::Ok(HttpResponse::ok().body(r#"{"a2a_servers":[]}"#).build());
        assert!(core.resolve(request.id.0, &encode(&empty)).is_empty());
        assert!(matches!(receive(&rx).effect, EffectFfi::Render(_)));
        let event = Event::Configs(ConfigsEvent::A2a(A2aConfigEvent::EditUrl {
            value: "https://remote.example.test".to_string(),
        }));
        assert!(core.update(&encode(&event)).is_empty());
        assert!(matches!(receive(&rx).effect, EffectFfi::Render(_)));
        let event = Event::Configs(ConfigsEvent::A2a(A2aConfigEvent::Create { base_url }));
        assert!(core.update(&encode(&event)).is_empty());
        assert!(matches!(receive(&rx).effect, EffectFfi::Render(_)));
        let request = receive(&rx);
        let EffectFfi::Http(http) = request.effect else {
            panic!("expected POST");
        };
        assert_eq!(http.method, "POST");
        assert_eq!(
            http.url,
            "http://localhost:5150/api/app_configs/a2a-servers"
        );
        let response = HttpResult::Ok(HttpResponse::ok().body("null").build());
        assert!(core.resolve(request.id.0, &encode(&response)).is_empty());
        assert!(matches!(receive(&rx).effect, EffectFfi::Render(_)));
        let refresh = receive(&rx);
        assert!(matches!(refresh.effect, EffectFfi::Http(request) if request.method == "GET"));
        let view: ViewModel = BincodeFfiFormat::deserialize(&core.view()).unwrap();
        assert!(view.a2a_config.saving && view.a2a_config.loading && view.a2a_config.created);
        assert!(core.resolve(refresh.id.0, &encode(&empty)).is_empty());
        assert!(matches!(receive(&rx).effect, EffectFfi::Render(_)));
        let view: ViewModel = BincodeFfiFormat::deserialize(&core.view()).unwrap();
        assert!(view.a2a_config.loaded && !view.a2a_config.saving && !view.a2a_config.loading);
        assert_eq!(
            view.a2a_config.notice.as_deref(),
            Some("已添加远程Agent配置。")
        );
    }

    #[test]
    fn native_mcp_roundtrip_omits_json_secrets_from_view_bytes() {
        let (tx, rx) = mpsc::channel();
        let core = CoreFfi::new(Arc::new(RecordingShell(tx)));
        let base_url = "http://localhost:5150".to_string();
        let get = Event::Configs(ConfigsEvent::Mcp(McpConfigEvent::Get {
            base_url: base_url.clone(),
        }));
        assert!(core.update(&encode(&get)).is_empty());
        assert!(matches!(receive(&rx).effect, EffectFfi::Render(_)));
        let request = receive(&rx);
        let empty = HttpResult::Ok(HttpResponse::ok().body(r#"{"mcp_servers":[]}"#).build());
        assert!(core.resolve(request.id.0, &encode(&empty)).is_empty());
        assert!(matches!(receive(&rx).effect, EffectFfi::Render(_)));
        let config = r#"{"mcpServers":{"demo":{"transport":"stdio","command":"node","env":{"TOKEN":"FFI_TEST_SECRET"}}}}"#;
        let edit = Event::Configs(ConfigsEvent::Mcp(McpConfigEvent::EditJson {
            value: config.to_string(),
        }));
        assert!(!format!("{edit:?}").contains("FFI_TEST_SECRET"));
        assert!(core.update(&encode(&edit)).is_empty());
        assert!(matches!(receive(&rx).effect, EffectFfi::Render(_)));
        let secret = b"FFI_TEST_SECRET";
        let bytes = core.view();
        assert!(!bytes.windows(secret.len()).any(|part| part == secret));
        let view: ViewModel = BincodeFfiFormat::deserialize(&bytes).unwrap();
        assert!(view.mcp_config.draft_present);
        let create = Event::Configs(ConfigsEvent::Mcp(McpConfigEvent::Create { base_url }));
        assert!(core.update(&encode(&create)).is_empty());
        assert!(matches!(receive(&rx).effect, EffectFfi::Render(_)));
        let request = receive(&rx);
        let EffectFfi::Http(http) = request.effect else {
            panic!("expected MCP POST");
        };
        assert_eq!(http.method, "POST");
        assert_eq!(
            http.url,
            "http://localhost:5150/api/app_configs/mcp-servers"
        );
        let body: serde_json::Value = serde_json::from_slice(&http.body).unwrap();
        assert_eq!(
            body["mcpServers"]["demo"]["env"]["TOKEN"],
            "FFI_TEST_SECRET"
        );
        let ack = HttpResult::Ok(HttpResponse::ok().body(r#"{"mcpServers":{"demo":{"transport":"stdio","enabled":true,"env":{"TOKEN":"FFI_TEST_SECRET"}}}}"#).build());
        assert!(core.resolve(request.id.0, &encode(&ack)).is_empty());
        assert!(matches!(receive(&rx).effect, EffectFfi::Render(_)));
        let refresh = receive(&rx);
        assert!(matches!(refresh.effect, EffectFfi::Http(request) if request.method == "GET"));
        let bytes = core.view();
        assert!(!bytes.windows(secret.len()).any(|part| part == secret));
        let view: ViewModel = BincodeFfiFormat::deserialize(&bytes).unwrap();
        assert!(view.mcp_config.saving && view.mcp_config.created);
        assert!(!view.mcp_config.draft_present);
        assert_eq!(view.mcp_config.servers[0].server_name, "demo");
        assert!(core.resolve(refresh.id.0, &encode(&empty)).is_empty());
        assert!(matches!(receive(&rx).effect, EffectFfi::Render(_)));
        let view: ViewModel = BincodeFfiFormat::deserialize(&core.view()).unwrap();
        assert!(!view.mcp_config.saving && !view.mcp_config.loading);
    }
}
