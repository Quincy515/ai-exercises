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

// Native effects use the same app::Effect wire schema as typegen and WASM.
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
    use crate::{Event, ViewModel, app::EffectFfi};

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
}
