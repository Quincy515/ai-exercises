import assert from "node:assert/strict";
import test from "node:test";
import { setImmediate as nextTurn } from "node:timers/promises";
import { Core } from "../dist/crux-tests/lib/crux/core.js";
import {
  a2aConfigEvents as events,
  configEvents as agentEvents,
  agentConfigFields,
  llmConfigEvents,
  llmConfigFields,
} from "../dist/crux-tests/features/configs/events.js";

const origin = "http://localhost:3000";
const listUrl = `${origin}/api/app_configs/a2a-servers`;
const serviceUrl = "https://agent.example";
const server = {
  id: "writer-agent",
  name: "测试写作助手",
  description: "只用于验证 A2A 配置流程",
  input_modes: ["text/plain"],
  output_modes: ["text/plain"],
  streaming: true,
  push_notifications: false,
  enabled: true,
};

async function harness(t) {
  const requests = [],
    errors = [],
    views = [];
  t.mock.method(
    globalThis,
    "fetch",
    (request) =>
      new Promise((resolve, reject) => {
        requests.push({ request, resolve, reject });
      }),
  );
  const core = new Core(
    (view) => views.push(view),
    (error) => errors.push(error),
  );
  t.after(() => core.dispose());
  await core.initialize();
  const reply = async (index, value, status = 200) => {
    requests[index].resolve(new Response(JSON.stringify(value), { status }));
    await nextTurn();
  };
  const load = async (servers = [server]) => {
    core.update(events.Get(origin));
    await reply(requests.length - 1, { a2a_servers: servers });
  };
  return {
    core,
    requests,
    errors,
    views,
    reply,
    load,
    state: () => core.view().a2a_config,
  };
}

test("A2A list decodes its envelope and invalid creation URLs never reach HTTP", async (t) => {
  const { core, requests, reply, state, errors } = await harness(t);
  core.update(events.Get(origin));
  core.update(events.Get(origin));
  assert.equal(requests.length, 1);
  assert.equal(requests[0].request.method, "GET");
  assert.equal(requests[0].request.url, listUrl);
  assert.equal(state().loading, true);
  assert.equal(state().loaded, false);
  await reply(0, { a2a_servers: [server] });
  assert.deepEqual(
    state().servers.map((item) => ({ ...item })),
    [server],
  );
  assert.equal(state().loaded, true);
  assert.equal(state().loading, false);
  for (const value of [
    "",
    "invalid-url",
    "ftp://agent.example",
    "https://user:password@agent.example",
  ]) {
    core.update(events.EditUrl(value));
    core.update(events.Create(origin));
    assert.equal(requests.length, 1);
    assert.ok(state().error);
    assert.equal(state().saving, false);
  }
  core.update(events.ResetDraft());
  assert.equal(state().draft_url, "");
  assert.equal(state().error, null);
  assert.equal(state().servers.length, 1);
  assert.deepEqual(errors, []);
});

test("A2A create accepts JSON null then keeps saving until its single refresh finishes", async (t) => {
  const { core, requests, reply, load, state, errors } = await harness(t);
  await load([]);
  core.update(events.EditUrl(serviceUrl));
  core.update(events.Create(origin));
  assert.equal(requests.length, 2);
  assert.equal(requests[1].request.method, "POST");
  assert.equal(requests[1].request.url, listUrl);
  assert.deepEqual(await requests[1].request.json(), { base_url: serviceUrl });
  const assertBusy = () => {
    const count = requests.length;
    core.update(events.Create(origin));
    core.update(events.Get(origin));
    core.update(events.SetEnabled(origin, server.id, false));
    core.update(events.Delete(origin, server.id));
    core.update(events.EditUrl("https://ignored.example"));
    core.update(events.ResetDraft());
    assert.equal(requests.length, count);
    assert.equal(state().saving, true);
  };
  assertBusy();
  assert.equal(state().draft_url, serviceUrl);
  await reply(1, null);
  assert.equal(requests.length, 3);
  assert.equal(requests[2].request.method, "GET");
  assert.equal(requests[2].request.url, listUrl);
  assert.equal(state().draft_url, "");
  assert.equal(state().created, true);
  assertBusy();
  await reply(2, { a2a_servers: [server] });
  assert.equal(state().saving, false);
  assert.equal(state().loading, false);
  assert.equal(state().servers[0].id, server.id);
  assert.equal(state().write_uncertain, false);
  assert.deepEqual(errors, []);
});

test("A2A enable and delete encode IDs as one path segment and revalidate after null", async (t) => {
  const { core, requests, reply, load, state } = await harness(t);
  const item = { ...server, id: "writer/中文 ?#id" };
  await load([item]);
  core.update(events.SetEnabled(origin, item.id, false));
  assert.equal(
    requests[1].request.url,
    `${listUrl}/${encodeURIComponent(item.id)}/enabled`,
  );
  assert.equal(requests[1].request.method, "POST");
  assert.deepEqual(await requests[1].request.json(), { enabled: false });
  await reply(1, null);
  assert.equal(state().servers[0].enabled, false);
  assert.equal(state().saving, true);
  assert.equal(requests[2].request.method, "GET");
  await reply(2, { a2a_servers: [{ ...item, enabled: false }] });
  assert.equal(state().saving, false);
  core.update(events.Delete(origin, item.id));
  assert.equal(
    requests[3].request.url,
    `${listUrl}/${encodeURIComponent(item.id)}/delete`,
  );
  assert.equal(requests[3].request.method, "POST");
  assert.equal(await requests[3].request.text(), "");
  await reply(3, null);
  assert.deepEqual(state().servers, []);
  assert.equal(state().saving, true);
  assert.equal(requests[4].request.method, "GET");
  await reply(4, { a2a_servers: [] });
  assert.equal(state().saving, false);
  assert.equal(state().error, null);
});

test("A2A a failed refresh after a confirmed create retries GET without a duplicate POST", async (t) => {
  const { core, requests, reply, load, state } = await harness(t);
  await load([]);
  core.update(events.EditUrl(serviceUrl));
  core.update(events.Create(origin));
  await reply(1, null);
  await reply(2, { message: "unavailable" }, 500);
  assert.equal(state().saving, false);
  assert.equal(state().loading, false);
  assert.equal(state().created, true);
  assert.equal(state().draft_url, "");
  assert.equal(state().write_uncertain, false);
  assert.ok(state().error);
  core.update(events.Get(origin));
  assert.deepEqual(
    requests.map(({ request }) => request.method),
    ["GET", "POST", "GET", "GET"],
  );
  await reply(3, { a2a_servers: [server] });
  assert.equal(state().error, null);
  assert.equal(state().servers[0].id, server.id);
});

for (const failure of [
  "network",
  "timeout",
  "malformed-success",
  "server-error",
]) {
  test(`A2A ${failure} makes the write uncertain and blocks more writes until a successful GET`, async (t) => {
    const { core, requests, reply, load, state, errors } = await harness(t);
    await load();
    core.update(events.EditUrl(serviceUrl));
    core.update(events.Create(origin));
    if (failure === "malformed-success")
      requests[1].resolve(new Response("not JSON"));
    else if (failure === "server-error")
      requests[1].resolve(
        new Response("write response failed", { status: 500 }),
      );
    else
      requests[1].reject(
        failure === "timeout"
          ? new DOMException("deadline", "TimeoutError")
          : new Error("offline"),
      );
    await nextTurn();
    assert.equal(state().saving, false);
    assert.equal(state().write_uncertain, true);
    assert.equal(state().draft_url, serviceUrl);
    assert.ok(state().error);
    const assertWritesBlocked = () => {
      const count = requests.length;
      core.update(events.Create(origin));
      core.update(events.SetEnabled(origin, server.id, false));
      core.update(events.Delete(origin, server.id));
      assert.equal(requests.length, count);
    };
    assertWritesBlocked();
    core.update(events.Get(origin));
    await reply(2, null, 500);
    assert.equal(state().write_uncertain, true);
    assertWritesBlocked();
    core.update(events.Get(origin));
    await reply(3, { a2a_servers: [server] });
    assert.equal(state().write_uncertain, false);
    assert.equal(state().error, null);
    assert.doesNotMatch(state().notice ?? "", /尚未确认/);
    core.update(events.SetEnabled(origin, server.id, false));
    assert.equal(requests.length, 5);
    await reply(4, null);
    await reply(5, { a2a_servers: [{ ...server, enabled: false }] });
    assert.equal(state().saving, false);
    assert.deepEqual(errors, []);
  });
}

test("A2A rejected writes preserve the list and draft for a manual retry", async (t) => {
  const { core, requests, reply, load, state } = await harness(t);
  await load();
  core.update(events.EditUrl(serviceUrl));
  core.update(events.Create(origin));
  await reply(1, { message: "invalid remote card" }, 422);
  assert.equal(state().draft_url, serviceUrl);
  assert.equal(state().write_uncertain, false);
  assert.equal(state().servers[0].enabled, true);
  assert.ok(state().error);
  assert.equal(state().saving, false);
  core.update(events.Delete(origin, server.id));
  await reply(2, null, 404);
  assert.equal(state().servers.length, 1);
  assert.equal(state().write_uncertain, false);
  core.update(events.SetEnabled(origin, server.id, false));
  await reply(3, null);
  await reply(4, { a2a_servers: [{ ...server, enabled: false }] });
  assert.equal(state().servers[0].enabled, false);
  assert.equal(state().error, null);
  assert.deepEqual(
    requests.map(({ request }) => request.method),
    ["GET", "POST", "POST", "POST", "GET"],
  );
});

test("A2A failures preserve Agent and LLM drafts in the shared Core", async (t) => {
  const { core, requests, reply, load, state, errors } = await harness(t);
  core.update(agentEvents.GetAgentConfig(origin));
  await reply(0, { max_iterations: 42, max_retries: 2, max_search_results: 8 });
  core.update(llmConfigEvents.Get(origin));
  await reply(1, {
    base_url: null,
    model_name: "test-model",
    temperature: null,
    max_tokens: null,
    api_key_configured: false,
  });
  core.update(
    agentEvents.EditAgentConfig(agentConfigFields.max_iterations, "43"),
  );
  core.update(llmConfigEvents.Edit(llmConfigFields.model_name, "draft-model"));
  const agent = core.view().agent_config,
    llm = core.view().llm_config;
  await load();
  core.update(events.Delete(origin, server.id));
  requests.at(-1).reject(new Error("A2A offline"));
  await nextTurn();
  assert.equal(state().write_uncertain, true);
  assert.deepEqual(core.view().agent_config, agent);
  assert.deepEqual(core.view().llm_config, llm);
  assert.deepEqual(errors, []);
});

test("disposing settings during the post-write refresh cancels it and ignores late renders", async (t) => {
  const { core, requests, reply, load, views, errors } = await harness(t);
  await load();
  core.update(events.Delete(origin, server.id));
  await reply(1, null);
  assert.equal(requests.length, 3);
  const signal = requests[2].request.signal;
  const count = views.length;
  core.dispose();
  assert.equal(signal.aborted, true);
  await reply(2, { a2a_servers: [] });
  assert.equal(views.length, count);
  assert.deepEqual(errors, []);
});
