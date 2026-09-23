import assert from "node:assert/strict";
import test from "node:test";
import { setImmediate as nextTurn } from "node:timers/promises";
import { Core } from "../dist/crux-tests/core.js";
import { cruxEvents as events } from "../dist/crux-tests/use-crux.js";

const body = (value) => JSON.stringify({ value, updated_at: 1672531200000 });

test(
  "Core initializes once, dispatches Rust events, and resolves HTTP then Time",
  { timeout: 3000 },
  async (t) => {
    let respond;
    let request;
    t.mock.method(globalThis, "fetch", (input) => {
      request = input;
      return new Promise((resolve) => {
        respond = resolve;
      });
    });
    const views = [];
    const errors = [];
    let synchronized;
    const confirmed = new Promise((resolve) => {
      synchronized = resolve;
    });
    const core = new Core(
      (view) => {
        views.push(view);
        if (view.confirmed) synchronized();
      },
      (error) => errors.push(error),
    );
    try {
      const init = core.initialize();
      assert.equal(core.initialize(), init);
      await init;
      assert.deepEqual(core.view(), { text: "0 (pending)", confirmed: false });
      core.update(events.Increment());
      assert.deepEqual(core.view(), { text: "1 (pending)", confirmed: false });
      assert.equal(request.method, "POST");
      assert.equal(request.url, "https://crux-counter.fly.dev/inc");
      respond(
        new Response(body(9), {
          headers: { "Content-Type": "application/json" },
        }),
      );
      await confirmed;
      assert.equal(core.view().text, "9 (2023-01-01 00:00:00 UTC)");
      assert.deepEqual(errors, []);
      core.update(events.Reset());
      assert.deepEqual(core.view(), { text: "0 (pending)", confirmed: false });
      assert.ok(views.length >= 4);
    } finally {
      core.dispose();
    }
  },
);

test("HTTP failure is reported and the Rust core remains usable", async (t) => {
  t.mock.method(globalThis, "fetch", async () => {
    throw new Error("offline");
  });
  const errors = [];
  const core = new Core(
    () => {},
    (error) => errors.push(error),
  );
  try {
    await core.initialize();
    core.update(events.Get());
    await nextTurn();
    assert.equal(errors.length, 1);
    assert.match(errors[0].message, /offline/);
    core.update(events.Reset());
    assert.equal(core.view().text, "0 (pending)");
  } finally {
    core.dispose();
  }
});

test(
  "SSE forwards split bytes to the Rust decoder and updates the view",
  { timeout: 3000 },
  async (t) => {
    const frame = new TextEncoder().encode(
      'data: {"value":42,"updated_at":null}\n\n',
    );
    t.mock.method(
      globalThis,
      "fetch",
      async () =>
        new Response(
          new ReadableStream({
            start(controller) {
              controller.enqueue(frame.slice(0, 15));
              controller.enqueue(frame.slice(15));
              controller.close();
            },
          }),
        ),
    );
    let updated;
    const update = new Promise((resolve) => {
      updated = resolve;
    });
    const errors = [];
    const core = new Core(
      (view) => {
        if (view.text === "42 (pending)") updated();
      },
      (error) => errors.push(error),
    );
    try {
      await core.initialize();
      core.update(events.StartWatch());
      await update;
      await nextTurn();
      assert.equal(core.view().text, "42 (pending)");
      assert.deepEqual(errors, []);
    } finally {
      core.dispose();
    }
  },
);

test("SSE failure closes the Rust stream and reports an error", async (t) => {
  t.mock.method(
    globalThis,
    "fetch",
    async () => new Response("unavailable", { status: 503 }),
  );
  const errors = [];
  const core = new Core(
    () => {},
    (error) => errors.push(error),
  );
  try {
    await core.initialize();
    core.update(events.StartWatch());
    await nextTurn();
    assert.equal(errors.length, 1);
    assert.match(errors[0].message, /503/);
    core.update(events.Reset());
    assert.equal(core.view().text, "0 (pending)");
  } finally {
    core.dispose();
  }
});

test("dispose during initialization prevents a late Core instance", async () => {
  const views = [];
  const core = new Core((view) => views.push(view));
  const loading = core.initialize();
  core.dispose();
  await loading;
  assert.equal(core.ready, false);
  assert.deepEqual(views, []);
  await assert.rejects(core.initialize(), /disposed/);
});

test("dispose aborts pending HTTP and suppresses late renders and errors", async (t) => {
  let signal;
  t.mock.method(globalThis, "fetch", (request) => {
    signal = request.signal;
    return new Promise((_, reject) => {
      signal.addEventListener(
        "abort",
        () => reject(new DOMException("cancelled", "AbortError")),
        { once: true },
      );
    });
  });
  const views = [];
  const errors = [];
  const core = new Core(
    (view) => views.push(view),
    (error) => errors.push(error),
  );
  await core.initialize();
  core.update(events.Get());
  core.dispose();
  core.dispose();
  await nextTurn();
  assert.equal(signal.aborted, true);
  assert.equal(views.length, 1);
  assert.deepEqual(errors, []);
  assert.equal(core.ready, false);
});
