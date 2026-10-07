import assert from "node:assert/strict";
import test from "node:test";
import { setImmediate as nextTurn } from "node:timers/promises";
import { Core } from "../dist/crux-tests/lib/crux/core.js";
import {
  mcpConfigEvents as events,
  configEvents as agentEvents,
  agentConfigFields,
  llmConfigEvents,
  llmConfigFields,
  a2aConfigEvents,
} from "../dist/crux-tests/features/configs/events.js";

const origin = "http://localhost:3000";
const listUrl = `${origin}/api/app_configs/mcp-servers`;
const secret = "fictional-mcp-test-secret-not-a-credential";
const server = {
  server_name: "tools",
  enabled: true,
  transport: "stdio",
  tools: ["read_file"],
};
const privateConfig = {
  transport: "stdio",
  enabled: true,
  command: "node",
  args: ["fixture.mjs", secret],
  env: { TEST_TOKEN: secret },
};
const draft = JSON.stringify({ mcpServers: { tools: privateConfig } });
const configResponse = (entries = { tools: privateConfig }) => ({
  mcpServers: entries,
});
const printable = (value) =>
  JSON.stringify(value, (_, item) =>
    typeof item === "bigint" ? String(item) : item,
  );

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
    await reply(requests.length - 1, { mcp_servers: servers });
  };
  const assertSafe = () => {
    assert.ok(views.every((view) => !printable(view).includes(secret)));
    assert.deepEqual(errors, []);
  };
  return {
    core,
    requests,
    errors,
    views,
    reply,
    load,
    assertSafe,
    state: () => core.view().mcp_config,
  };
}

test("MCP GET decodes public metadata and discards extra private response fields", async (t) => {
  const { core, requests, reply, state, assertSafe } = await harness(t);
  core.update(events.Get(origin));
  core.update(events.Get(origin));
  assert.equal(requests.length, 1);
  assert.equal(requests[0].request.method, "GET");
  assert.equal(requests[0].request.url, listUrl);
  assert.equal(state().loading, true);
  await reply(0, {
    mcp_servers: [
      { ...server, env: { TOKEN: secret }, headers: { Authorization: secret } },
    ],
  });
  assert.deepEqual(
    state().servers.map((item) => ({ ...item })),
    [server],
  );
  assert.equal(state().loaded, true);
  assert.equal(state().loading, false);
  assert.equal(state().draft_present, false);
  assertSafe();
});

test("MCP JSON validation prevents HTTP writes and keeps private drafts out of views", async (t) => {
  const { core, requests, load, state, assertSafe } = await harness(t);
  await load();
  for (const value of [
    `{"mcpServers":${secret}}`,
    "[]",
    "{}",
    '{"mcpServers":{}}',
    '{"mcpServers":{"bad":{"transport":"sse","url":"https://mcp.example"}}}',
    '{"mcpServers":{"bad":{"transport":"stdio"}}}',
    '{"mcpServers":{"bad":{"transport":"streamable_http"}}}',
    '{"mcpServers":{"bad":{"transport":"stdio","command":"node","env":[]}}}',
    '{"mcpServers":{"bad":{"transport":"streamable_http","url":"https://mcp.example","headers":[]}}}',
  ]) {
    core.update(events.EditJson(value));
    assert.equal(state().draft_present, true);
    core.update(events.Create(origin));
    assert.equal(
      requests.length,
      1,
      "invalid JSON must not create a write effect",
    );
    assert.equal(state().saving, false);
    assert.ok(state().error);
    assertSafe();
  }
  core.update(events.ResetDraft());
  assert.equal(state().draft_present, false);
  assert.equal(state().error, null);
  assertSafe();
});

test("MCP batch upsert forwards both transports and confirms safe metadata before revalidation", async (t) => {
  const { core, requests, reply, load, state, assertSafe } = await harness(t);
  const retained = { ...server, server_name: "retained" };
  await load([server, retained]);
  const httpConfig = {
    transport: "streamable_http",
    enabled: true,
    url: "https://mcp.example/endpoint",
    headers: { Authorization: `Bearer ${secret}` },
  };
  const payload = { mcpServers: { tools: privateConfig, remote: httpConfig } };
  core.update(events.EditJson(JSON.stringify(payload)));
  assert.equal(state().draft_present, true);
  assertSafe();
  core.update(events.Create(origin));
  assert.equal(requests.length, 2);
  assert.equal(requests[1].request.method, "POST");
  assert.equal(requests[1].request.url, listUrl);
  assert.match(
    requests[1].request.headers.get("content-type"),
    /application\/json/,
  );
  const sent = await requests[1].request.json();
  assert.deepEqual(Object.keys(sent.mcpServers).sort(), ["remote", "tools"]);
  assert.equal(sent.mcpServers.tools.transport, "stdio");
  assert.deepEqual(sent.mcpServers.tools.env, privateConfig.env);
  assert.deepEqual(sent.mcpServers.tools.args, privateConfig.args);
  assert.equal(sent.mcpServers.remote.transport, "streamable_http");
  assert.deepEqual(sent.mcpServers.remote.headers, httpConfig.headers);
  const assertBusy = () => {
    const count = requests.length;
    core.update(events.Create(origin));
    core.update(events.Get(origin));
    core.update(events.SetEnabled(origin, server.server_name, false));
    core.update(events.Delete(origin, server.server_name));
    core.update(events.EditJson("{}"));
    core.update(events.ResetDraft());
    assert.equal(requests.length, count);
    assert.equal(state().saving, true);
  };
  assertBusy();
  await reply(
    1,
    configResponse({
      tools: privateConfig,
      remote: httpConfig,
      retained: privateConfig,
    }),
  );
  assert.equal(state().created, true);
  assert.equal(state().draft_present, false);
  assert.equal(requests.length, 3);
  assert.equal(requests[2].request.method, "GET");
  assert.equal(requests[2].request.url, listUrl);
  assertBusy();
  assertSafe();
  const remote = {
    server_name: "remote",
    enabled: true,
    transport: "streamable_http",
    tools: ["search"],
  };
  await reply(2, { mcp_servers: [server, retained, remote] });
  assert.equal(state().saving, false);
  assert.equal(state().loading, false);
  assert.equal(state().write_uncertain, false);
  assert.deepEqual(
    state().servers.map((item) => ({ ...item })),
    [server, retained, remote],
  );
  assertSafe();
});

test("MCP enable and delete encode names once and confirm their returned configuration", async (t) => {
  const { core, requests, reply, load, state, assertSafe } = await harness(t);
  const item = { ...server, server_name: "tools/中文 ?#name" };
  await load([item]);
  core.update(events.SetEnabled(origin, item.server_name, false));
  assert.equal(
    requests[1].request.url,
    `${listUrl}/${encodeURIComponent(item.server_name)}/enabled`,
  );
  assert.equal(requests[1].request.method, "POST");
  assert.deepEqual(await requests[1].request.json(), { enabled: false });
  await reply(
    1,
    configResponse({
      [item.server_name]: { ...privateConfig, enabled: false },
    }),
  );
  assert.equal(state().saving, true);
  assert.equal(state().servers[0].enabled, false);
  assertSafe();
  await reply(2, { mcp_servers: [{ ...item, enabled: false }] });
  core.update(events.Delete(origin, item.server_name));
  assert.equal(
    requests[3].request.url,
    `${listUrl}/${encodeURIComponent(item.server_name)}/delete`,
  );
  assert.equal(requests[3].request.method, "POST");
  assert.equal(await requests[3].request.text(), "");
  await reply(3, configResponse({}));
  assert.equal(state().saving, true);
  assert.deepEqual(state().servers, []);
  await reply(4, { mcp_servers: [] });
  assert.equal(state().saving, false);
  assert.equal(state().error, null);
  assertSafe();
});

test("MCP confirmed writes survive a failed refresh and only GET is retried", async (t) => {
  const { core, requests, reply, load, state, assertSafe } = await harness(t);
  await load();
  core.update(events.EditJson(draft));
  core.update(events.Create(origin));
  await reply(1, configResponse());
  await reply(2, { message: secret }, 500);
  assert.equal(state().created, true);
  assert.equal(state().draft_present, false);
  assert.equal(state().saving, false);
  assert.equal(state().write_uncertain, false);
  assert.ok(state().error);
  core.update(events.Get(origin));
  assert.deepEqual(
    requests.map(({ request }) => request.method),
    ["GET", "POST", "GET", "GET"],
  );
  await reply(3, { mcp_servers: [server] });
  assert.equal(state().error, null);
  assertSafe();
});

for (const failure of [
  "network",
  "timeout",
  "server-error",
  "null",
  "malformed",
  "unrelated-config",
]) {
  test(`MCP ${failure} blocks repeated writes until the user refreshes the uncertain result`, async (t) => {
    const { core, requests, reply, load, state, assertSafe } = await harness(t);
    await load();
    core.update(events.EditJson(draft));
    core.update(events.Create(origin));
    if (failure === "network")
      requests[1].reject(new Error(`offline ${secret}`));
    else if (failure === "timeout")
      requests[1].reject(new DOMException("deadline", "TimeoutError"));
    else if (failure === "server-error")
      requests[1].resolve(new Response(secret, { status: 500 }));
    else if (failure === "null") await reply(1, null);
    else if (failure === "malformed")
      requests[1].resolve(new Response(`invalid JSON ${secret}`));
    else await reply(1, configResponse({ unrelated: privateConfig }));
    await nextTurn();
    assert.equal(state().saving, false);
    assert.equal(state().created, false);
    assert.equal(state().draft_present, true);
    assert.equal(state().write_uncertain, true);
    assert.ok(state().error);
    assert.equal(
      requests.length,
      2,
      "an unconfirmed response must not start the successful-write refresh",
    );
    const assertWritesBlocked = () => {
      const count = requests.length;
      core.update(events.Create(origin));
      core.update(events.SetEnabled(origin, server.server_name, false));
      core.update(events.Delete(origin, server.server_name));
      assert.equal(requests.length, count);
    };
    assertWritesBlocked();
    core.update(events.Get(origin));
    await reply(2, null, 500);
    assert.equal(state().write_uncertain, true);
    assertWritesBlocked();
    core.update(events.Get(origin));
    await reply(3, { mcp_servers: [server] });
    assert.equal(state().write_uncertain, false);
    assert.equal(state().error, null);
    core.update(events.SetEnabled(origin, server.server_name, false));
    assert.equal(requests.length, 5);
    await reply(
      4,
      configResponse({ tools: { ...privateConfig, enabled: false } }),
    );
    await reply(5, { mcp_servers: [{ ...server, enabled: false }] });
    assertSafe();
  });
}

test("MCP contradictory enable and delete responses never become confirmed success", async (t) => {
  const { core, requests, reply, load, state, assertSafe } = await harness(t);
  await load();
  core.update(events.SetEnabled(origin, server.server_name, false));
  await reply(1, configResponse());
  assert.equal(state().write_uncertain, true);
  assert.equal(state().servers[0].enabled, true);
  assert.equal(requests.length, 2);
  core.update(events.Get(origin));
  await reply(2, { mcp_servers: [server] });
  core.update(events.Delete(origin, server.server_name));
  await reply(3, configResponse());
  assert.equal(state().write_uncertain, true);
  assert.equal(state().servers.length, 1);
  assert.equal(requests.length, 4);
  assertSafe();
});

test("MCP known validation errors preserve the private draft for a manual retry", async (t) => {
  const { core, requests, reply, load, state, assertSafe } = await harness(t);
  await load();
  core.update(events.EditJson(draft));
  core.update(events.Create(origin));
  await reply(1, { message: secret }, 422);
  assert.equal(state().write_uncertain, false);
  assert.equal(state().draft_present, true);
  assert.equal(state().saving, false);
  assert.ok(state().error);
  core.update(events.Create(origin));
  assert.equal(
    (await requests[2].request.json()).mcpServers.tools.env.TEST_TOKEN,
    secret,
  );
  await reply(2, configResponse());
  await reply(3, { mcp_servers: [server] });
  assert.equal(state().draft_present, false);
  assert.equal(state().error, null);
  assertSafe();
});

test("MCP failures preserve Agent, LLM and A2A drafts in the shared Core", async (t) => {
  const { core, requests, reply, load, state, assertSafe } = await harness(t);
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
  core.update(a2aConfigEvents.Get(origin));
  await reply(2, { a2a_servers: [] });
  core.update(
    agentEvents.EditAgentConfig(agentConfigFields.max_iterations, "43"),
  );
  core.update(llmConfigEvents.Edit(llmConfigFields.model_name, "draft-model"));
  core.update(a2aConfigEvents.EditUrl("https://agent.example"));
  const before = core.view();
  await load();
  core.update(events.Delete(origin, server.server_name));
  requests.at(-1).reject(new Error(`MCP offline ${secret}`));
  await nextTurn();
  assert.equal(state().write_uncertain, true);
  for (const resource of ["agent_config", "llm_config", "a2a_config"]) {
    assert.deepEqual(core.view()[resource], before[resource]);
  }
  assertSafe();
});

test("disposing settings cancels MCP I/O and ignores a late private write response", async (t) => {
  const { core, requests, reply, load, views, assertSafe } = await harness(t);
  await load();
  core.update(events.EditJson(draft));
  core.update(events.Create(origin));
  const signal = requests[1].request.signal;
  const count = views.length;
  core.dispose();
  assert.equal(signal.aborted, true);
  await reply(1, configResponse());
  assert.equal(views.length, count);
  assert.equal(requests.length, 2);
  assertSafe();
});
