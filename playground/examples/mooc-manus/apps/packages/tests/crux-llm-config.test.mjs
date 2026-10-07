import assert from "node:assert/strict";
import test from "node:test";
import { setImmediate as nextTurn } from "node:timers/promises";
import { Core } from "../dist/crux-tests/lib/crux/core.js";
import {
  configEvents as agentEvents,
  agentConfigFields,
  llmConfigEvents as events,
  llmConfigFields as fields,
} from "../dist/crux-tests/features/configs/events.js";

const origin = "http://localhost:3000";
const llmUrl = `${origin}/api/app_configs/llm`;
const config = {
  base_url: "https://provider.example/v1",
  model_name: "test-model",
  temperature: 0.5,
  max_tokens: 4096,
  api_key_configured: true,
};
const agentConfig = {
  max_iterations: 42,
  max_retries: 2,
  max_search_results: 8,
};
const json = (value) => new Response(JSON.stringify(value));
const printable = (view) =>
  JSON.stringify(view, (_, value) =>
    typeof value === "bigint" ? String(value) : value,
  );

test("LLM GET and POST preserve nullable fields and omit an unchanged API key", async (t) => {
  const requests = [];
  let respond;
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
  try {
    await core.initialize();
    core.update(events.Get(origin));
    core.update(events.Get(origin));
    assert.equal(requests.length, 1);
    assert.equal(requests[0].url, llmUrl);
    assert.equal(requests[0].method, "GET");
    assert.equal(core.view().llm_config.loading, true);
    respond(
      json({
        base_url: null,
        model_name: null,
        temperature: null,
        max_tokens: null,
        api_key_configured: false,
      }),
    );
    await nextTurn();
    const loaded = core.view().llm_config;
    assert.equal(loaded.data.base_url, null);
    assert.equal(loaded.data.model_name, null);
    assert.equal(loaded.data.temperature, null);
    assert.equal(loaded.data.max_tokens, null);
    assert.equal(loaded.data.api_key_configured, false);
    assert.deepEqual(
      { ...loaded.draft },
      {
        base_url: "",
        model_name: "",
        temperature: "",
        max_tokens: "",
      },
    );
    assert.equal(loaded.can_save, false);
    core.update(events.Edit(fields.model_name, "new-model"));
    core.update(events.Save(origin));
    assert.equal(requests.length, 2);
    assert.equal(requests[1].url, llmUrl);
    assert.equal(requests[1].method, "POST");
    assert.match(requests[1].headers.get("content-type"), /application\/json/);
    assert.deepEqual(await requests[1].json(), {
      base_url: null,
      model_name: "new-model",
      temperature: null,
      max_tokens: null,
    });
    respond(
      json({
        base_url: null,
        model_name: "confirmed-model",
        temperature: null,
        max_tokens: null,
        api_key_configured: false,
      }),
    );
    await nextTurn();
    const saved = core.view().llm_config;
    assert.equal(saved.data.model_name, "confirmed-model");
    assert.equal(saved.draft.model_name, "confirmed-model");
    assert.equal(saved.saved, true);
    assert.equal(saved.saving, false);
    assert.equal(saved.dirty, false);
    assert.equal(saved.can_save, false);
    assert.equal(saved.api_key_changed, false);
    assert.deepEqual(errors, []);
  } finally {
    core.dispose();
  }
});

test("LLM invalid drafts stay local and reset restores server-confirmed values", async (t) => {
  const requests = [];
  t.mock.method(globalThis, "fetch", async (request) => {
    requests.push(request);
    return json(config);
  });
  const core = new Core(() => {});
  try {
    await core.initialize();
    core.update(events.Get(origin));
    await nextTurn();
    for (const [field, value] of [
      [fields.base_url, "not a URL"],
      [fields.temperature, "3"],
      [fields.temperature, "NaN"],
      [fields.max_tokens, "2.5"],
      [fields.max_tokens, "-1"],
    ]) {
      core.update(events.Edit(field, value));
      core.update(events.Save(origin));
      assert.equal(requests.length, 1, "invalid values must never reach HTTP");
      assert.ok(core.view().llm_config.error);
      assert.equal(core.view().llm_config.saving, false);
      assert.equal(core.view().llm_config.dirty, true);
      core.update(events.Get(origin));
      assert.equal(requests.length, 1, "refresh must preserve the dirty draft");
      core.update(events.Reset());
      assert.equal(core.view().llm_config.error, null);
      assert.equal(core.view().llm_config.dirty, false);
    }
    core.update(events.Edit(fields.api_key, "fictional-discarded-key"));
    assert.equal(core.view().llm_config.api_key_changed, true);
    core.update(events.Reset());
    assert.equal(core.view().llm_config.api_key_changed, false);
    assert.equal(core.view().llm_config.can_save, false);
    assert.deepEqual(
      { ...core.view().llm_config.draft },
      {
        base_url: config.base_url,
        model_name: config.model_name,
        temperature: "0.5",
        max_tokens: "4096",
      },
    );
  } finally {
    core.dispose();
  }
});

test("LLM secrets only enter the request and failed saves retain drafts for explicit retry", async (t) => {
  const secret = "fictional-llm-test-key-not-a-credential";
  const requests = [];
  let respond, reject;
  t.mock.method(globalThis, "fetch", (request) => {
    requests.push(request);
    if (request.method === "GET")
      return Promise.resolve(json({ ...config, api_key: secret }));
    return new Promise((resolve, fail) => {
      respond = resolve;
      reject = fail;
    });
  });
  const errors = [],
    views = [];
  const core = new Core(
    (view) => views.push(view),
    (error) => errors.push(error),
  );
  try {
    await core.initialize();
    core.update(events.Get(origin));
    await nextTurn();
    core.update(events.Edit(fields.model_name, "updated-model"));
    core.update(events.Edit(fields.api_key, secret));
    assert.equal(core.view().llm_config.api_key_changed, true);
    assert.ok(!printable(core.view()).includes(secret));

    for (const failure of [500, 422, "network", "timeout"]) {
      core.update(events.Save(origin));
      const requestCount = requests.length;
      core.update(events.Save(origin));
      core.update(events.Get(origin));
      core.update(events.Edit(fields.model_name, "ignored-during-save"));
      core.update(events.Edit(fields.api_key, "ignored-during-save"));
      core.update(events.Reset());
      assert.equal(requests.length, requestCount);
      assert.equal(core.view().llm_config.saving, true);
      assert.equal(core.view().llm_config.can_save, false);
      assert.deepEqual(await requests.at(-1).json(), {
        base_url: config.base_url,
        model_name: "updated-model",
        temperature: config.temperature,
        max_tokens: config.max_tokens,
        api_key: secret,
      });
      if (failure === "network") reject(new Error(`offline ${secret}`));
      else if (failure === "timeout")
        reject(new DOMException("request timed out", "TimeoutError"));
      else respond(new Response(`rejected ${secret}`, { status: failure }));
      await nextTurn();
      const failed = core.view().llm_config;
      assert.equal(failed.saving, false);
      assert.equal(failed.data.model_name, config.model_name);
      assert.equal(failed.draft.model_name, "updated-model");
      assert.equal(failed.api_key_changed, true);
      assert.equal(failed.can_save, true);
      assert.ok(failed.error);
      assert.ok(!printable(failed).includes(secret));
      assert.deepEqual(
        errors,
        [],
        "business failures must not reach global errors",
      );
    }

    core.update(events.Save(origin));
    respond(
      json({
        ...config,
        model_name: "server-confirmed-model",
        api_key: secret,
      }),
    );
    await nextTurn();
    const saved = core.view().llm_config;
    assert.equal(saved.data.model_name, "server-confirmed-model");
    assert.equal(saved.draft.model_name, "server-confirmed-model");
    assert.equal(saved.saved, true);
    assert.equal(saved.api_key_changed, false);
    assert.equal(saved.can_save, false);
    assert.equal(saved.error, null);
    assert.ok(views.every((view) => !printable(view).includes(secret)));

    core.update(events.Edit(fields.model_name, "next-model"));
    core.update(events.Save(origin));
    assert.equal(Object.hasOwn(await requests.at(-1).json(), "api_key"), false);
    respond(json({ ...config, model_name: "next-model" }));
    await nextTurn();
  } finally {
    core.dispose();
  }
});

test("Agent and LLM loading, drafts and network errors remain isolated in one Core", async (t) => {
  const pending = new Map();
  t.mock.method(
    globalThis,
    "fetch",
    (request) =>
      new Promise((resolve, reject) => {
        pending.set(request.url, { resolve, reject });
      }),
  );
  const errors = [];
  const core = new Core(
    () => {},
    (error) => errors.push(error),
  );
  const agentUrl = `${origin}/api/app_configs/agent`;
  try {
    await core.initialize();
    core.update(agentEvents.GetAgentConfig(origin));
    core.update(events.Get(origin));
    assert.equal(core.view().agent_config.loading, true);
    assert.equal(core.view().llm_config.loading, true);
    pending.get(llmUrl).reject(new Error("LLM offline"));
    pending.get(agentUrl).resolve(json(agentConfig));
    await nextTurn();
    assert.ok(core.view().llm_config.error);
    assert.equal(core.view().llm_config.loading, false);
    assert.equal(core.view().agent_config.error, null);
    assert.equal(core.view().agent_config.data.max_iterations, 42n);
    core.update(
      agentEvents.EditAgentConfig(agentConfigFields.max_iterations, "43"),
    );
    core.update(events.Get(origin));
    pending.get(llmUrl).resolve(json(config));
    await nextTurn();
    core.update(events.Edit(fields.model_name, "independent-model"));
    assert.equal(core.view().agent_config.draft.max_iterations, "43");
    core.update(agentEvents.ResetAgentConfig());
    assert.equal(core.view().llm_config.draft.model_name, "independent-model");
    core.update(agentEvents.GetAgentConfig(origin));
    pending.get(agentUrl).reject(new Error("Agent offline"));
    await nextTurn();
    assert.ok(core.view().agent_config.error);
    assert.equal(core.view().llm_config.error, null);
    assert.equal(core.view().llm_config.draft.model_name, "independent-model");
    assert.equal(core.view().llm_config.can_save, true);
    assert.deepEqual(errors, []);
  } finally {
    core.dispose();
  }
});

test("disposing a shared settings Core cancels LLM save without late UI updates", async (t) => {
  let signal;
  t.mock.method(globalThis, "fetch", (request) => {
    if (request.method === "GET") return Promise.resolve(json(config));
    signal = request.signal;
    return new Promise((_, reject) =>
      signal.addEventListener(
        "abort",
        () => reject(new DOMException("cancelled", "AbortError")),
        { once: true },
      ),
    );
  });
  const views = [],
    errors = [];
  const core = new Core(
    (view) => views.push(view),
    (error) => errors.push(error),
  );
  try {
    await core.initialize();
    core.update(events.Get(origin));
    await nextTurn();
    core.update(events.Edit(fields.model_name, "updated-model"));
    core.update(events.Save(origin));
    const count = views.length;
    core.dispose();
    await nextTurn();
    assert.equal(signal.aborted, true);
    assert.equal(views.length, count);
    assert.deepEqual(errors, []);
  } finally {
    core.dispose();
  }
});
