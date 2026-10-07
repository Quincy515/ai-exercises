#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { allowedConfigWrite, ConfigWriteJournal, requireConfigWriteAuthorization } from './config-write.mjs';
import { startMcpFixture } from './mcp-http-fixture.mjs';
import { finalizeMcpFixture } from './mcp-flows.mjs';
import { allowedMcpWrite, mcpFixtureFingerprint, mcpFixturePayload, mcpFixtureUrl, mcpPath, mcpRecoveryStatus,
  mcpWriteEvidence, readMcpList, readMcpWriteResponse, sameMcpList, validMcpFixture } from './mcp-policy.mjs';

const nonce = '12345678-1234-4234-9234-123456789abc';
const fixture = { schema: 'mcp-http-fixture/v1', nonce, port: 45678, serverName: `verification-mcp-${nonce}`, toolName: `verify_${nonce.replaceAll('-', '')}` };
const existing = { server_name: 'original', transport: 'stdio', enabled: false, tools: [] };
const own = { server_name: fixture.serverName, transport: 'streamable_http', enabled: true, tools: [fixture.toolName] };
const expected = action => ({ fixture, baseline: [existing], action, createAttempted: false, ownedFingerprint: mcpFixtureFingerprint(fixture), enabled: false, revision: 'initial' });
const write = overrides => ({ feature: 'mcp-write', allowConfigWrite: true, baseUrl: 'http://127.0.0.1:4317',
  url: 'http://127.0.0.1:4317' + mcpPath, method: 'POST', payload: mcpFixturePayload(fixture), expected: expected('create'), ...overrides });

test('MCP writes require fresh CLI authorization before backend or fixture work', () => {
  assert.throws(() => requireConfigWriteAuthorization(['mcp-write']), /mcp-write requires explicit/);
  const script = fileURLToPath(new URL('./verify.mjs', import.meta.url));
  for (const verb of ['launch', 'run', 'drive']) {
    const result = spawnSync(process.execPath, [script, verb, '--features', 'mcp-write'], { encoding: 'utf8' });
    assert.equal(result.status, 1); assert.match(result.stderr, /mcp-write requires explicit --allow-config-write true/);
  }
});

test('MCP fixture descriptor derives only a nonce-scoped loopback URL', () => {
  assert.equal(validMcpFixture(fixture), true);
  assert.equal(mcpFixtureUrl(fixture), `http://127.0.0.1:45678/${nonce}/mcp`);
  for (const delta of [{ schema: 'wrong' }, { nonce: 'unsafe/path' }, { port: 80 }, { port: 65536 }, { serverName: 'original' }, { toolName: 'exec' }]) assert.equal(validMcpFixture({ ...fixture, ...delta }), false);
});

test('MCP list parser only exposes public metadata and rejects duplicate names', () => {
  assert.deepEqual(readMcpList(JSON.stringify({ mcp_servers: [{ ...existing, url: 'SECRET-URL', env: { token: 'SECRET-ENV' } }] })), [existing]);
  assert.throws(() => readMcpList(JSON.stringify({ mcp_servers: [existing, existing] })), /duplicate/);
  assert.throws(() => readMcpList(JSON.stringify({ mcp_servers: [{ ...existing, transport: 'sse' }] })), /metadata/);
  assert.throws(() => readMcpList('SECRET-RAW'), error => !error.message.includes('SECRET'));
});

test('MCP create is single-attempt and supports only the precise opted-in endpoint', () => {
  assert.equal(allowedConfigWrite(write()), true);
  for (const delta of [{ feature: 'mcp' }, { allowConfigWrite: false }, { method: 'PUT' },
    { url: 'https://elsewhere.test' + mcpPath }, { url: write().url + '?token=hidden' },
    { expected: { ...expected('create'), createAttempted: true } }, { expected: { ...expected('create'), baseline: [own] } }]) assert.equal(allowedConfigWrite(write(delta)), false);
});

test('MCP create and update deny stdio, remote URLs, credentials and unrelated names', () => {
  const input = mcpFixturePayload(fixture).mcpServers[fixture.serverName];
  for (const delta of [{ transport: 'stdio', command: 'echo' }, { url: 'https://model.example/mcp' },
    { env: { SECRET: 'never' } }, { headers: { Authorization: 'never' } }, { args: ['never'] },
    { command: 'echo' }, { other: 'never' }, { enabled: false }, { description: 'different' }]) {
    assert.equal(allowedMcpWrite(write({ payload: { mcpServers: { [fixture.serverName]: { ...input, ...delta } } } })), false);
  }
  assert.equal(allowedConfigWrite(write({ payload: { mcpServers: { ...mcpFixturePayload(fixture).mcpServers, original: input } } })), false);
  assert.equal(allowedConfigWrite(write({ payload: { ...mcpFixturePayload(fixture), extra: true } })), false);
  assert.equal(allowedConfigWrite(write({ payload: { mcpServers: { [fixture.serverName]: { ...input, env: null, headers: null, args: null, command: null } } } })), true);
});

test('MCP same-name update needs confirmed ownership and the expected revision', () => {
  const update = { expected: { ...expected('update'), revision: 'updated' }, payload: mcpFixturePayload(fixture, 'updated') };
  assert.equal(allowedConfigWrite(write(update)), true);
  for (const ownedFingerprint of [null, 'wrong']) assert.equal(allowedConfigWrite(write({ ...update, expected: { ...update.expected, ownedFingerprint } })), false);
  assert.equal(allowedConfigWrite(write({ ...update, payload: mcpFixturePayload(fixture) })), false);
});

test('MCP toggle and delete guard preserve every baseline name', () => {
  for (const action of ['enabled', 'delete']) {
    const options = { url: write().url + '/' + fixture.serverName + '/' + action, expected: expected(action),
      payload: action === 'enabled' ? { enabled: false } : undefined, body: null };
    assert.equal(allowedConfigWrite(write(options)), true);
    assert.equal(allowedConfigWrite(write({ ...options, url: write().url + '/original/' + action })), false);
    assert.equal(allowedConfigWrite(write({ ...options, expected: { ...expected(action), baseline: [own] } })), false);
    assert.equal(allowedConfigWrite(write({ ...options, expected: { ...expected(action), ownedFingerprint: null } })), false);
  }
  assert.equal(allowedConfigWrite(write({ url: write().url + '/' + fixture.serverName + '/delete', expected: expected('delete'), body: '{}' })), false);
  assert.equal(allowedConfigWrite(write({ url: write().url + '/' + fixture.serverName + '/enabled', expected: expected('enabled'), payload: { enabled: true } })), false);
});

test('full MCP write responses are reduced to safe metadata while own URL is checked in memory', () => {
  const raw = { mcpServers: { original: { transport: 'stdio', enabled: false, command: 'SECRET-CMD', args: ['SECRET-ARGS'], env: { token: 'SECRET-ENV' }, headers: { key: 'SECRET-HEADER' }, url: 'SECRET-URL' },
    ...mcpFixturePayload(fixture).mcpServers } };
  const safe = readMcpWriteResponse(JSON.stringify(raw), { fixture, action: 'create' });
  assert.deepEqual(safe.mcp_servers, [{ server_name: 'original', transport: 'stdio', enabled: false }, { server_name: fixture.serverName, transport: 'streamable_http', enabled: true }]);
  assert.equal(safe.ownedFingerprint, mcpFixtureFingerprint(fixture));
  assert.ok(!JSON.stringify(safe).includes('SECRET') && !JSON.stringify(safe).includes('127.0.0.1'));
  raw.mcpServers[fixture.serverName].url = 'SECRET-CONFLICT';
  assert.throws(() => readMcpWriteResponse(JSON.stringify(raw), { fixture, action: 'update' }), error => !error.message.includes('SECRET'));
});

test('MCP confirmation checks same-name update, disabled state, and owned deletion', () => {
  const raw = mcpFixturePayload(fixture, 'updated');
  raw.mcpServers[fixture.serverName].enabled = false;
  readMcpWriteResponse(JSON.stringify(raw), { fixture, action: 'enabled', revision: 'updated', enabled: false });
  assert.throws(() => readMcpWriteResponse(JSON.stringify(raw), { fixture, action: 'enabled', enabled: false }), /changed/);
  assert.throws(() => readMcpWriteResponse(JSON.stringify(raw), { fixture, action: 'delete' }), /still contains/);
  assert.deepEqual(readMcpWriteResponse('{"mcpServers":{}}', { fixture, action: 'delete' }), { mcp_servers: [] });
});

test('MCP POST journal evidence excludes even controlled configuration fields', () => {
  const safe = mcpWriteEvidence(expected('create'));
  assert.deepEqual(safe, { action: 'create', server_name: fixture.serverName, transport: 'streamable_http', enabled: true });
  const journal = new ConfigWriteJournal();
  journal.begin({ url: write().url, payload: safe });
  assert.ok(!JSON.stringify(journal.entries).includes(mcpFixtureUrl(fixture)));
  for (const key of ['env', 'headers', 'args', 'command', 'description']) assert.ok(!JSON.stringify(safe).includes(key));
});

test('unknown MCP writes require manual recovery even when the list equals baseline', () => {
  const journal = new ConfigWriteJournal();
  journal.unknown(journal.begin({ url: write().url, payload: mcpWriteEvidence(expected('create')) }), 'timeout');
  const result = mcpRecoveryStatus({ journal, baseline: [existing], current: [existing], fixture, ownedFingerprint: null, createAttempted: true, deleteConfirmed: false });
  assert.equal(result.decision, 'outcome-unknown'); assert.equal(result.ownedName, null);
  assert.equal(result.manualCleanupRequired, true); assert.equal(result.visibleListRestored, undefined);
});

test('MCP delete restoration is limited to public metadata and preserves prior ordering flexibility', () => {
  const journal = new ConfigWriteJournal(); journal.complete(journal.begin({ url: write().url, payload: {} }), 200);
  const input = { journal, baseline: [existing], current: [existing], fixture, ownedFingerprint: mcpFixtureFingerprint(fixture), createAttempted: true, deleteConfirmed: true };
  assert.equal(mcpRecoveryStatus(input).visibleListRestored, true);
  assert.equal(mcpRecoveryStatus(input).manualCleanupRequired, false);
  assert.equal(sameMcpList([existing, own], [own, existing]), true);
  assert.equal(sameMcpList([existing], [{ ...existing, enabled: true }]), false);
});

test('local MCP fixture initializes, lists tools, rejects invocation, sanitizes proof, and restarts', async t => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'mooc-mcp-fixture-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  let active;
  try {
    active = await startMcpFixture(directory); const saved = active.descriptor;
    const call = data => fetch(mcpFixtureUrl(saved), { method: 'POST', headers: { 'Content-Type': 'application/json', Connection: 'close' }, body: JSON.stringify(data) });
    const initialize = await call({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-03-26' } });
    assert.equal(initialize.status, 200); assert.equal((await initialize.json()).result.protocolVersion, '2025-03-26');
    assert.equal((await call({ jsonrpc: '2.0', method: 'notifications/initialized' })).status, 202);
    const tools = await (await call({ jsonrpc: '2.0', id: 2, method: 'tools/list' })).json();
    assert.equal(tools.result.tools[0].name, saved.toolName);
    assert.equal((await call({ jsonrpc: '2.0', id: 3, method: 'tools/call', params: { secret: 'NEVER-RECORD' } })).status, 403);
    assert.equal((await call({ jsonrpc: '2.0', id: 4, method: 'NEVER-RECORD', params: {} })).status, 403);
    const proof = fs.readFileSync(path.join(directory, 'fixture-requests.jsonl'), 'utf8');
    assert.ok(!proof.includes('NEVER-RECORD') && !proof.includes('params'));
    assert.ok(!fs.readFileSync(path.join(directory, 'fixture.json'), 'utf8').includes('http://'));
    await active.close(); active = await startMcpFixture(directory, saved);
    assert.equal(mcpFixtureUrl(active.descriptor), mcpFixtureUrl(saved));
    assert.equal((await call({ jsonrpc: '2.0', id: 5, method: 'tools/list' })).status, 200);
  } finally { await active?.close(); }
});

test('fixture close drains an entered partial POST once and its final gate rejects the late request', { timeout: 5000 }, async t => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'mooc-mcp-drain-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const createServer = http.createServer.bind(http);
  let entered;
  const requestEntered = new Promise(resolve => { entered = resolve; });
  t.mock.method(http, 'createServer', (...args) => {
    const server = createServer(...args);
    server.once('request', entered);
    return server;
  });
  const active = await startMcpFixture(directory);
  const socket = net.connect(active.descriptor.port, '127.0.0.1');
  const closed = new Promise(resolve => { socket.once('close', resolve); });
  socket.on('error', () => {});
  try {
    socket.write(`POST /${active.descriptor.nonce}/mcp HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: 50\r\nConnection: close\r\n\r\n{`);
    await requestEntered;
    assert.deepEqual(active.observations, [], 'Handler is waiting for the remaining JSON bytes');
    const first = active.close(), concurrent = active.close();
    assert.equal(first, concurrent, 'Concurrent close calls share one completion boundary');
    await Promise.all([first, concurrent]);
    assert.equal(active.observations.length, 1);
    assert.equal(active.observations[0].operation, 'invalid-json');
    assert.equal(active.observations[0].accepted, false);
    const result = { status: 'passed', configCleanup: {} };
    await finalizeMcpFixture(active, result);
    assert.equal(result.status, 'failed');
    assert.equal(result.configCleanup.fixtureStopped, true);
    assert.deepEqual(result.fixtureSafetyGate, { checkedAfterClose: true, rejectedRequests: 1, toolsListed: false });
    const lifecycle = fs.readFileSync(path.join(directory, 'fixture-lifecycle.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
    assert.equal(lifecycle.filter(entry => entry.stoppedAt).length, 1);
    const snapshot = JSON.stringify(active.observations);
    fs.rmSync(directory, { recursive: true });
    await closed;
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(JSON.stringify(active.observations), snapshot, 'No handler can record evidence after close returns');
    assert.equal(fs.existsSync(directory), false);
  } finally { socket.destroy(); await active.close(); }
});

test('fixture shutdown failure is recorded as failed with sanitized cleanup evidence', async () => {
  const result = { status: 'passed', configCleanup: { visibleListRestored: true } };
  await finalizeMcpFixture({ observations: [], close: async () => { throw new Error('SECRET-DIAGNOSTIC'); } }, result);
  assert.equal(result.status, 'failed');
  assert.equal(result.configCleanup.fixtureStopped, false);
  assert.deepEqual(result.fixtureSafetyGate, { checkedAfterClose: false, closeFailed: true });
  assert.ok(result.cleanupError && !result.cleanupError.includes('SECRET'));
});
