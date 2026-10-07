import type { HttpRequest, HttpResult } from "shared_types/app.js";
import {
  HttpResponse,
  HttpHeader,
  httpErrorIo,
  httpErrorTimeout,
  httpErrorUrl,
  httpResultErr,
  httpResultOk,
} from "shared_types/app.js";

export async function request(
  { url, method, headers, body }: HttpRequest,
  {
    signal,
    timeoutMs = 30_000,
  }: { signal?: AbortSignal; timeoutMs?: number } = {},
): Promise<HttpResult> {
  try {
    new URL(url);
  } catch (error) {
    return httpResultErr(httpErrorUrl(String(error)));
  }

  // 普通 HTTP 请求有截止时间，包含响应体读取；SSE 使用独立的流式适配器。
  const controller = new AbortController();
  const abort = () => controller.abort(signal?.reason);
  if (signal?.aborted) abort();
  else signal?.addEventListener("abort", abort, { once: true });
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(
      new Request(url, {
        method,
        headers: headers.map((header): [string, string] => [
          header.name,
          header.value,
        ]),
        body: ["GET", "HEAD"].includes(method.toUpperCase())
          ? undefined
          : new Uint8Array(body),
        signal: controller.signal,
      }),
    );

    const responseHeaders = Array.from(
      response.headers.entries(),
      ([name, value]) => new HttpHeader(name, value),
    );

    // HTTP 状态码和原始响应体交给 Rust 判断，传输失败转换为 HttpError。
    return httpResultOk(
      new HttpResponse(
        response.status,
        responseHeaders,
        new Uint8Array(await response.arrayBuffer()),
      ),
    );
  } catch (error) {
    if (
      controller.signal.aborted ||
      (error instanceof Error &&
        ["AbortError", "TimeoutError"].includes(error.name))
    ) {
      return httpResultErr(httpErrorTimeout());
    }
    return httpResultErr(
      httpErrorIo(error instanceof Error ? error.message : String(error)),
    );
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", abort);
  }
}
