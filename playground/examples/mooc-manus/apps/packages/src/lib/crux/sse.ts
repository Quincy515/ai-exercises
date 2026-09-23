import type { SseRequest, SseResponse } from "shared_types/app.js";
import { sseResponseDone, sseResponseChunk } from "shared_types/app.js";

export async function* request(
  { url }: SseRequest,
  { signal }: { signal?: AbortSignal } = {},
): AsyncGenerator<SseResponse> {
  const response = await fetch(
    new Request(url, {
      headers: { Accept: "text/event-stream" },
      signal,
    }),
  );

  const reader = response.body?.getReader();
  let completed = false;
  try {
    if (!response.ok) {
      throw new Error(`SSE request failed: HTTP ${response.status}`);
    }

    if (!reader) {
      // 无响应体 — 视为流结束
      yield sseResponseDone();
      return;
    }

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        completed = true;
        yield sseResponseDone();
        return;
      }
      // 原始网络块交给 Rust 的持续解码器，保留跨块的消息和 UTF-8 字节。
      yield sseResponseChunk(Array.from(value));
    }
  } finally {
    if (reader && !completed) {
      try {
        await reader.cancel();
      } catch {
        // 已出错的流也可能拒绝取消，继续释放锁并保留原始错误。
      }
    }
    reader?.releaseLock();
  }
}
