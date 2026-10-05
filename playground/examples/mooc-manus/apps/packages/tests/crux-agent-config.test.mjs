import assert from "node:assert/strict";
import test from "node:test";
import { setImmediate as nextTurn } from "node:timers/promises";
import { Core } from "../dist/crux-tests/core.js";
import { cruxEvents as events } from "../dist/crux-tests/use-crux.js";

test(
  "Agent config crosses real WASM, HTTP Shell and Bincode, then recovers from failure",
  { timeout: 5000 },
  async (t) => {
    let respond;
    const requests = [];
    t.mock.method(globalThis, "fetch", (request) => {
      requests.push(request);
      return new Promise((resolve) => {
        respond = resolve;
      });
    });
    const errors = [];
    const core = new Core(
      () => {},
      (error) => errors.push(error),
    );
    const refresh = () =>
      core.update(events.GetAgentConfig("http://localhost:3000"));
    try {
      await core.initialize();
      refresh();
      refresh();
      assert.equal(requests.length, 1);
      assert.equal(requests[0].method, "GET");
      assert.equal(
        requests[0].url,
        "http://localhost:3000/api/app_configs/agent",
      );
      assert.equal(core.view().agent_config.loading, true);
      respond(
        new Response(
          '{"max_iterations":42,"max_retries":2,"max_search_results":8}',
        ),
      );
      await nextTurn();
      const state = core.view().agent_config;
      assert.equal(state.loading, false);
      assert.equal(state.error, null);
      assert.equal(state.data.max_iterations, 42n);
      assert.equal(state.data.max_retries, 2n);
      assert.equal(state.data.max_search_results, 8n);

      refresh();
      respond(new Response("unavailable", { status: 500 }));
      await nextTurn();
      assert.equal(core.view().agent_config.loading, false);
      assert.match(core.view().agent_config.error, /HTTP 500/);
      assert.deepEqual(core.view().agent_config.data, state.data);

      refresh();
      respond(
        new Response(
          '{"max_iterations":99,"max_retries":4,"max_search_results":12}',
        ),
      );
      await nextTurn();
      assert.equal(core.view().agent_config.error, null);
      assert.equal(core.view().agent_config.data.max_iterations, 99n);
      assert.deepEqual(errors, []);
    } finally {
      core.dispose();
    }
  },
);

test("closing Agent settings aborts HTTP and suppresses late UI updates", async (t) => {
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
  core.update(events.GetAgentConfig("http://localhost:3000"));
  const count = views.length;
  core.dispose();
  await nextTurn();
  assert.equal(signal.aborted, true);
  assert.equal(views.length, count);
  assert.deepEqual(errors, []);
});
