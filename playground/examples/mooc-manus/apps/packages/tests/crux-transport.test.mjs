import assert from "node:assert/strict";
import test from "node:test";

import { request as http } from "../dist/crux-tests/lib/crux/http.js";
import { request as sse } from "../dist/crux-tests/lib/crux/sse.js";

const httpRequest = (overrides = {}) => ({
  method: "GET",
  url: "https://example.com/api",
  headers: [],
  body: new Uint8Array(),
  ...overrides,
});

test("HTTP preserves binary bodies, headers, status, and abort signal", async (t) => {
  const controller = new AbortController();
  t.mock.method(globalThis, "fetch", async (request) => {
    assert.equal(request.url, "https://example.com/api");
    assert.equal(request.method, "POST");
    assert.equal(request.headers.get("x-request"), "value");
    assert.deepEqual(
      new Uint8Array(await request.arrayBuffer()),
      new Uint8Array([0, 128, 255]),
    );
    controller.abort();
    assert.equal(request.signal.aborted, true);
    return new Response(new Uint8Array([255, 0, 128]), {
      status: 503,
      headers: { "x-response": "reply" },
    });
  });

  const result = await http(
    httpRequest({
      method: "POST",
      headers: [{ name: "x-request", value: "value" }],
      body: new Uint8Array([0, 128, 255]),
    }),
    { signal: controller.signal },
  );

  assert.equal(result.kind, "Ok");
  assert.equal(result.value.status, 503);
  assert.deepEqual(
    result.value.headers.map(({ name, value }) => [name, value]),
    [["x-response", "reply"]],
  );
  assert.deepEqual(result.value.body, new Uint8Array([255, 0, 128]));
});

test("HTTP omits bodies for GET and HEAD", async (t) => {
  const requests = [];
  t.mock.method(globalThis, "fetch", async (request) => {
    requests.push(request);
    return new Response(null, { status: 204 });
  });

  for (const method of ["get", "HEAD"]) {
    const result = await http(
      httpRequest({ method, body: new Uint8Array([42]) }),
    );
    assert.equal(result.kind, "Ok");
    assert.equal(result.value.status, 204);
    assert.equal(result.value.body.byteLength, 0);
  }
  assert.equal(requests.length, 2);
  assert.ok(requests.every((request) => request.body === null));
});

test("HTTP reports invalid URLs without fetching", async (t) => {
  const fetch = t.mock.method(globalThis, "fetch", async () => {
    throw new Error("fetch should not run");
  });

  const result = await http(httpRequest({ url: "not a valid URL" }));
  assert.equal(result.kind, "Err");
  assert.equal(result.value.kind, "Url");
  assert.equal(fetch.mock.callCount(), 0);
});

test("HTTP converts fetch and response-body failures to IO errors", async (t) => {
  const fetch = t.mock.method(globalThis, "fetch", async () => {
    throw new TypeError("network unavailable");
  });
  assert.deepEqual(await http(httpRequest()), {
    kind: "Err",
    value: { kind: "Io", value: "network unavailable" },
  });

  fetch.mock.mockImplementation(async () => ({
    status: 200,
    headers: new Headers(),
    arrayBuffer: async () => {
      throw new Error("body interrupted");
    },
  }));
  assert.deepEqual(await http(httpRequest()), {
    kind: "Err",
    value: { kind: "Io", value: "body interrupted" },
  });
});

test("HTTP converts abort and timeout failures to Timeout", async (t) => {
  const fetch = t.mock.method(globalThis, "fetch", async () => {
    throw new DOMException("cancelled", "AbortError");
  });
  const expected = { kind: "Err", value: { kind: "Timeout" } };
  assert.deepEqual(await http(httpRequest()), expected);

  fetch.mock.mockImplementation(async () => {
    throw new DOMException("expired", "TimeoutError");
  });
  assert.deepEqual(await http(httpRequest()), expected);

  const controller = new AbortController();
  controller.abort(new Error("cancelled by caller"));
  fetch.mock.mockImplementation(async () => {
    throw controller.signal.reason;
  });
  assert.deepEqual(
    await http(httpRequest(), { signal: controller.signal }),
    expected,
  );
});

test("HTTP deadline aborts hanging requests and response bodies", async (t) => {
  const waitForAbort = (signal) =>
    new Promise((_, reject) => {
      signal.addEventListener("abort", () => reject(signal.reason), {
        once: true,
      });
    });
  const fetch = t.mock.method(globalThis, "fetch", (request) =>
    waitForAbort(request.signal),
  );
  const expected = { kind: "Err", value: { kind: "Timeout" } };
  assert.deepEqual(await http(httpRequest(), { timeoutMs: 5 }), expected);

  fetch.mock.mockImplementation(async (request) => ({
    status: 200,
    headers: new Headers(),
    arrayBuffer: () => waitForAbort(request.signal),
  }));
  assert.deepEqual(await http(httpRequest(), { timeoutMs: 5 }), expected);
});

test("SSE forwards raw network chunks and emits Done at EOF", async (t) => {
  const bytes = new TextEncoder().encode('data: {"text":"你好"}\n\n');
  const chunks = [bytes.slice(0, 17), bytes.slice(17)];
  const stream = new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(chunk);
      controller.close();
    },
  });
  const controller = new AbortController();
  t.mock.method(globalThis, "fetch", async (request) => {
    assert.equal(request.url, "https://example.com/events");
    assert.equal(request.headers.get("accept"), "text/event-stream");
    controller.abort();
    assert.equal(request.signal.aborted, true);
    return new Response(stream);
  });

  const responses = [];
  for await (const response of sse(
    { url: "https://example.com/events" },
    { signal: controller.signal },
  )) {
    responses.push(response);
  }
  assert.deepEqual(responses, [
    ...chunks.map((chunk) => ({ kind: "Chunk", value: Array.from(chunk) })),
    { kind: "Done" },
  ]);
  assert.equal(stream.locked, false);
});

test("SSE emits Done for an empty response body", async (t) => {
  t.mock.method(
    globalThis,
    "fetch",
    async () => new Response(null, { status: 204 }),
  );
  const responses = [];
  for await (const response of sse({ url: "https://example.com/events" }))
    responses.push(response);
  assert.deepEqual(responses, [{ kind: "Done" }]);
});

test("SSE cancels and releases the reader when the consumer returns early", async (t) => {
  let cancellations = 0;
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(new Uint8Array([1, 2]));
    },
    cancel() {
      cancellations += 1;
    },
  });
  t.mock.method(globalThis, "fetch", async () => new Response(stream));

  const responses = sse({ url: "https://example.com/events" });
  assert.deepEqual(await responses.next(), {
    done: false,
    value: { kind: "Chunk", value: [1, 2] },
  });
  assert.equal(stream.locked, true);
  await responses.return();
  assert.equal(cancellations, 1);
  assert.equal(stream.locked, false);
});

test("SSE releases an errored reader and preserves the read failure", async (t) => {
  const failure = new Error("stream interrupted");
  let cancellations = 0;
  let releases = 0;
  t.mock.method(globalThis, "fetch", async () => ({
    ok: true,
    body: {
      getReader: () => ({
        read: async () => {
          throw failure;
        },
        cancel: async () => {
          cancellations += 1;
          throw failure;
        },
        releaseLock: () => {
          releases += 1;
        },
      }),
    },
  }));

  await assert.rejects(
    sse({ url: "https://example.com/events" }).next(),
    (error) => error === failure,
  );
  assert.equal(cancellations, 1);
  assert.equal(releases, 1);
});

test("SSE cancels an HTTP error response body", async (t) => {
  let cancellations = 0;
  const stream = new ReadableStream({
    cancel() {
      cancellations += 1;
    },
  });
  t.mock.method(
    globalThis,
    "fetch",
    async () => new Response(stream, { status: 503 }),
  );

  await assert.rejects(
    sse({ url: "https://example.com/events" }).next(),
    /503/,
  );
  assert.equal(cancellations, 1);
  assert.equal(stream.locked, false);
});

test("SSE propagates HTTP, network, and abort errors", async (t) => {
  const fetch = t.mock.method(
    globalThis,
    "fetch",
    async () => new Response(null, { status: 503 }),
  );
  await assert.rejects(
    sse({ url: "https://example.com/events" }).next(),
    /503/,
  );

  for (const failure of [
    new TypeError("network unavailable"),
    new DOMException("cancelled", "AbortError"),
  ]) {
    fetch.mock.mockImplementation(async () => {
      throw failure;
    });
    await assert.rejects(
      sse({ url: "https://example.com/events" }).next(),
      (error) => error === failure,
    );
  }
});
