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
  { signal }: { signal?: AbortSignal } = {},
): Promise<HttpResult> {
  try {
    new URL(url);
  } catch (error) {
    return httpResultErr(httpErrorUrl(String(error)));
  }

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
        signal,
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
      signal?.aborted ||
      (error instanceof Error &&
        ["AbortError", "TimeoutError"].includes(error.name))
    ) {
      return httpResultErr(httpErrorTimeout());
    }
    return httpResultErr(
      httpErrorIo(error instanceof Error ? error.message : String(error)),
    );
  }
}
