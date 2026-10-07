import assert from "node:assert/strict";
import test from "node:test";
import { setImmediate as nextTurn } from "node:timers/promises";
import { Core } from "../dist/crux-tests/lib/crux/core.js";
import {
  configEvents as events,
  agentConfigFields,
} from "../dist/crux-tests/features/configs/events.js";

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

test(
  "draft validation and POST save cross WASM with retry and server-confirmed data",
  { timeout: 5000 },
  async (t) => {
    const requests = [];
    let respond;
    t.mock.method(globalThis, "fetch", (request) => {
      requests.push(request);
      if (request.method === "GET")
        return Promise.resolve(
          new Response(
            '{"max_iterations":42,"max_retries":2,"max_search_results":8}',
          ),
        );
      return new Promise((resolve) => {
        respond = resolve;
      });
    });
    const errors = [];
    const core = new Core(
      () => {},
      (error) => errors.push(error),
    );
    const save = () =>
      core.update(events.SaveAgentConfig("http://localhost:3000"));
    try {
      await core.initialize();
      core.update(events.GetAgentConfig("http://localhost:3000"));
      await nextTurn();
      core.update(
        events.EditAgentConfig(agentConfigFields.max_iterations, "2.5"),
      );
      save();
      assert.equal(requests.length, 1, "invalid drafts must never reach fetch");
      assert.match(core.view().agent_config.error, /整数/);
      assert.equal(core.view().agent_config.draft.max_iterations, "2.5");
      core.update(events.ResetAgentConfig());
      assert.equal(core.view().agent_config.draft.max_iterations, "42");
      assert.equal(core.view().agent_config.dirty, false);

      core.update(
        events.EditAgentConfig(agentConfigFields.max_iterations, "43"),
      );
      save();
      save();
      core.update(events.GetAgentConfig("http://localhost:3000"));
      core.update(
        events.EditAgentConfig(agentConfigFields.max_iterations, "44"),
      );
      core.update(events.ResetAgentConfig());
      assert.equal(requests.length, 2);
      assert.equal(
        requests[1].url,
        "http://localhost:3000/api/app_configs/agent",
      );
      assert.equal(requests[1].method, "POST");
      assert.match(
        requests[1].headers.get("content-type"),
        /application\/json/,
      );
      assert.deepEqual(await requests[1].json(), {
        max_iterations: 43,
        max_retries: 2,
        max_search_results: 8,
      });
      assert.equal(core.view().agent_config.saving, true);
      assert.equal(core.view().agent_config.can_save, false);
      assert.equal(core.view().agent_config.draft.max_iterations, "43");

      respond(new Response("unavailable", { status: 500 }));
      await nextTurn();
      assert.match(core.view().agent_config.error, /HTTP 500/);
      assert.equal(core.view().agent_config.saving, false);
      assert.equal(core.view().agent_config.data.max_iterations, 42n);
      assert.equal(core.view().agent_config.draft.max_iterations, "43");
      assert.equal(core.view().agent_config.can_save, true);

      save();
      respond(
        new Response(
          '{"max_iterations":44,"max_retries":2,"max_search_results":8}',
        ),
      );
      await nextTurn();
      const saved = core.view().agent_config;
      assert.equal(saved.data.max_iterations, 44n);
      assert.equal(saved.draft.max_iterations, "44");
      assert.equal(saved.saved, true);
      assert.equal(saved.dirty, false);
      assert.equal(saved.can_save, false);
      assert.equal(saved.error, null);
      assert.deepEqual(errors, []);
    } finally {
      core.dispose();
    }
  },
);

test("disposing a Core during save cancels I/O and suppresses late rendering", async (t) => {
  let signal;
  t.mock.method(globalThis, "fetch", (request) => {
    if (request.method === "GET")
      return Promise.resolve(
        new Response(
          '{"max_iterations":42,"max_retries":2,"max_search_results":8}',
        ),
      );
    signal = request.signal;
    return new Promise((_, reject) => {
      signal.addEventListener(
        "abort",
        () => reject(new DOMException("cancelled", "AbortError")),
        { once: true },
      );
    });
  });
  const views = [],
    errors = [];
  const core = new Core(
    (view) => views.push(view),
    (error) => errors.push(error),
  );
  await core.initialize();
  core.update(events.GetAgentConfig("http://localhost:3000"));
  await nextTurn();
  core.update(events.EditAgentConfig(agentConfigFields.max_retries, "4"));
  core.update(events.SaveAgentConfig("http://localhost:3000"));
  const count = views.length;
  core.dispose();
  await nextTurn();
  assert.equal(signal.aborted, true);
  assert.equal(views.length, count);
  assert.deepEqual(errors, []);
});
