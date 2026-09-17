use std::{convert::From, future};

use async_sse::{Event as SseEvent, decode};
use crux_core::{Request, capability::Operation, command::StreamBuilder};
use facet::Facet;
use futures::{Stream, StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

#[derive(Facet, Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct SseRequest {
    pub url: String,
}

#[derive(Facet, Serialize, Deserialize, Debug, PartialEq, Eq)]
#[repr(C)]
pub enum SseResponse {
    Chunk(Vec<u8>),
    Done,
}

impl SseResponse {
    #[must_use]
    pub fn is_done(&self) -> bool {
        matches!(self, SseResponse::Done)
    }
}

impl Operation for SseRequest {
    type Output = SseResponse;
}

pub fn get<Effect, Event, T>(
    url: impl Into<String>,
) -> StreamBuilder<Effect, Event, impl Stream<Item = T>>
where
    Effect: From<Request<SseRequest>> + Send + 'static,
    Event: Send + 'static,
    T: Send + DeserializeOwned,
{
    let url = url.into();

    StreamBuilder::new(|ctx| {
        let chunks = ctx
            .stream_from_shell(SseRequest { url })
            .take_while(|response| future::ready(!response.is_done()))
            .map(|response| {
                let SseResponse::Chunk(data) = response else {
                    unreachable!()
                };

                Ok::<_, std::io::Error>(data)
            });

        decode(chunks.into_async_read()).filter_map(|sse_event| async {
            sse_event.ok().and_then(|event| match event {
                SseEvent::Message(msg) => serde_json::from_slice(msg.data()).ok(),
                SseEvent::Retry(_) => None, // do we need to worry about this?
            })
        })
    })
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{SseResponse, get};
    use crate::Effect;

    #[test]
    fn get_preserves_json_and_utf8_across_every_chunk_boundary() {
        let frame = "data: {\"value\":42,\"text\":\"你好\"}\n\n".as_bytes();

        for split in 1..frame.len() {
            let mut command = get::<Effect, Value, Value>("https://example.com/sse")
                .then_send(std::convert::identity);
            let Effect::ServerSentEvents(mut request) = command.expect_one_effect() else {
                panic!("expected an SSE request");
            };

            request
                .resolve(SseResponse::Chunk(frame[..split].to_vec()))
                .unwrap();
            assert!(command.events().next().is_none(), "split at {split}");

            request
                .resolve(SseResponse::Chunk(frame[split..].to_vec()))
                .unwrap();
            assert_eq!(
                command.events().collect::<Vec<_>>(),
                vec![json!({ "value": 42, "text": "你好" })],
                "split at {split}"
            );

            request.resolve(SseResponse::Done).unwrap();
            assert!(command.events().next().is_none());
            assert!(command.is_done());
        }
    }

    #[test]
    fn get_emits_complete_frames_and_discards_incomplete_frame_on_done() {
        let mut command = get::<Effect, Value, Value>("https://example.com/sse")
            .then_send(std::convert::identity);
        let Effect::ServerSentEvents(mut request) = command.expect_one_effect() else {
            panic!("expected an SSE request");
        };

        request
            .resolve(SseResponse::Chunk(
                b"data: {\"value\":1}\n\ndata: {\"value\":".to_vec(),
            ))
            .unwrap();
        assert_eq!(
            command.events().collect::<Vec<_>>(),
            vec![json!({ "value": 1 })]
        );

        request
            .resolve(SseResponse::Chunk(
                b"2}\n\ndata: {\"value\":3}\n\ndata: {\"value\":4}".to_vec(),
            ))
            .unwrap();
        assert_eq!(
            command.events().collect::<Vec<_>>(),
            vec![json!({ "value": 2 }), json!({ "value": 3 })]
        );

        request.resolve(SseResponse::Done).unwrap();
        assert!(command.events().next().is_none());
        assert!(command.is_done());
    }
}
