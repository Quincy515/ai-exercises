#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { allowedConfigWrite, ConfigWriteJournal, requireConfigWriteAuthorization } from './config-write.mjs';
import { a2aPath, a2aRecoveryStatus, createA2aOwnership, findOwnedA2a, readA2aList, sameVisibleA2a, validFixtureIdentity } from './a2a-policy.mjs';
import { nativeBackendVerdict, startCardFixture } from './a2a-card-fixture.mjs';

const nonce = '12345678-1234-4234-9234-123456789abc';
const fixture = { nonce, baseUrl: `http://127.0.0.1:45678/${nonce}`, name: `Verification Agent ${nonce}`, description: `Local controlled Agent Card ${nonce}` };
const existing = { id: '11111111-1111-4111-9111-111111111111', name: 'Existing', description: 'Original service', input_modes: ['text'], output_modes: ['text'], streaming: true, push_notifications: false, enabled: true };
const owned = { ...existing, id: '22222222-2222-4222-9222-222222222222', name: fixture.name, description: fixture.description, streaming: false };
const expected = action => ({ action, fixture, baseline: [existing], owned, knownId: owned.id, enabled: false, createAttempted: false });
const write = overrides => ({ feature: 'a2a-write', allowConfigWrite: true, baseUrl: 'http://127.0.0.1:4317',
  url: 'http://127.0.0.1:4317' + a2aPath, method: 'POST', payload: { base_url: fixture.baseUrl }, expected: expected('create'), ...overrides });

test('A2A write requires explicit authorization in every CLI entry', () => {
  assert.throws(() => requireConfigWriteAuthorization(['a2a-write']), /a2a-write requires explicit/);
  const script = fileURLToPath(new URL('./verify.mjs', import.meta.url));
  for (const verb of ['launch', 'run', 'drive']) {
    const result = spawnSync(process.execPath, [script, verb, '--features', 'a2a-write'], { encoding: 'utf8' });
    assert.equal(result.status, 1);
    assert.match(result.stderr, /a2a-write requires explicit --allow-config-write true/);
  }
});

test('A2A create allows only this local nonce URL and one explicit attempt', () => {
  assert.equal(allowedConfigWrite(write()), true);
  for (const overrides of [{ feature: 'a2a' }, { allowConfigWrite: false }, { method: 'GET' },
    { payload: { base_url: 'https://remote-agent.test' } }, { payload: { base_url: fixture.baseUrl, enabled: true } },
    { expected: { ...expected('create'), createAttempted: true } },
    { url: 'http://localhost:4317' + a2aPath }, { url: 'http://127.0.0.1:4317' + a2aPath + '?extra=1' }]) {
    assert.equal(allowedConfigWrite(write(overrides)), false);
  }
});

test('A2A identity requires exact local origin, UUID nonce and matching card labels', () => {
  assert.equal(validFixtureIdentity(fixture), true);
  for (const altered of [{ ...fixture, baseUrl: `http://localhost:45678/${nonce}` },
    { ...fixture, baseUrl: fixture.baseUrl + '?token=secret' }, { ...fixture, baseUrl: `https://127.0.0.1:45678/${nonce}` },
    { ...fixture, nonce: 'guess' }, { ...fixture, description: 'generic test agent' }]) assert.equal(validFixtureIdentity(altered), false);
});

test('A2A list parser whitelists public fields and rejects malformed/duplicate IDs', () => {
  assert.deepEqual(readA2aList(JSON.stringify({ a2a_servers: [{ ...existing, base_url: 'hidden', credential: 'secret' }] })), [existing]);
  assert.throws(() => readA2aList(JSON.stringify({ a2a_servers: [existing, existing] })), /duplicate IDs/);
  assert.throws(() => readA2aList(JSON.stringify({ a2a_servers: [{ ...existing, enabled: 'true' }] })), /public contract/);
});

test('A2A ownership needs unique name and description plus a new server-issued ID', () => {
  assert.deepEqual(findOwnedA2a([existing, owned], [existing], fixture), owned);
  assert.equal(findOwnedA2a([existing, { ...owned, description: 'another record' }], [existing], fixture), null);
  assert.equal(findOwnedA2a([{ ...owned, id: existing.id }], [existing], fixture), null);
  assert.throws(() => findOwnedA2a([owned, { ...owned, id: '33333333-3333-4333-9333-333333333333' }], [existing], fixture), /ambiguous/);
  assert.throws(() => findOwnedA2a([{ ...owned, id: '../../original' }], [existing], fixture), /server-issued UUID/);
});

test('A2A enable guard rejects original records and altered enabled values', () => {
  const options = { url: `http://127.0.0.1:4317${a2aPath}/${owned.id}/enabled`, payload: { enabled: false }, expected: expected('enabled') };
  assert.equal(allowedConfigWrite(write(options)), true);
  assert.equal(allowedConfigWrite(write({ ...options, payload: { enabled: true } })), false);
  assert.equal(allowedConfigWrite(write({ ...options, expected: { ...expected('enabled'), baseline: [existing, owned] } })), false);
  assert.equal(allowedConfigWrite(write({ ...options, expected: { ...expected('enabled'), owned: existing } })), false);
});

test('A2A delete requires the proved owned ID and an empty body', () => {
  const options = { url: `http://127.0.0.1:4317${a2aPath}/${owned.id}/delete`, payload: null, body: null, expected: expected('delete') };
  assert.equal(allowedConfigWrite(write(options)), true);
  assert.equal(allowedConfigWrite(write({ ...options, body: '' })), true);
  for (const body of ['null', '{}', ' ', '{"id":"original"}']) assert.equal(allowedConfigWrite(write({ ...options, body })), false);
  assert.equal(allowedConfigWrite(write({ ...options, url: `http://127.0.0.1:4317${a2aPath}/${existing.id}/delete` })), false);
});

test('unknown or invisible A2A create keeps supervised cleanup required and never invents an ID', () => {
  const journal = new ConfigWriteJournal();
  const entry = journal.begin({ url: write().url, payload: write().payload });
  journal.unknown(entry, 'POST timeout');
  const invisible = a2aRecoveryStatus({ journal, baseline: [existing], current: [existing], fixture, createAttempted: true, deleteConfirmed: false });
  assert.equal(invisible.decision, 'outcome-unknown');
  assert.equal(invisible.ownedId, null);
  assert.equal(invisible.manualCleanupRequired, true);
  assert.equal(invisible.visibleListRestored, undefined);
  const located = a2aRecoveryStatus({ journal, baseline: [existing], current: [existing, owned], fixture, createAttempted: true, deleteConfirmed: false });
  assert.equal(located.ownedId, owned.id);
  assert.equal(located.manualCleanupRequired, true);
});

test('confirmed A2A delete proves only the visible list and unknown deletion remains unresolved', () => {
  const journal = new ConfigWriteJournal();
  const entry = journal.begin({ url: write().url, payload: write().payload });
  journal.complete(entry, 200);
  const input = { journal, baseline: [existing], current: [existing], fixture, knownId: owned.id, createAttempted: true, deleteConfirmed: true };
  assert.equal(a2aRecoveryStatus(input).visibleListRestored, true);
  assert.equal(a2aRecoveryStatus(input).manualCleanupRequired, false);
  assert.equal(sameVisibleA2a([existing, owned], [owned, existing]), true);
  assert.equal(sameVisibleA2a([{ ...existing, enabled: false }], [existing]), false);
  journal.unknown(journal.begin({ url: write().url, payload: null }), 'Deletion outcome unknown');
  assert.equal(a2aRecoveryStatus(input).visibleListRestored, undefined);
  assert.equal(a2aRecoveryStatus(input).manualCleanupRequired, true);
});

test('native backend prerequisite rejects containers, wrong checkout and ambiguous listener ownership', () => {
  const input = { pids: [96833], executable: '/tmp/build/server-cli', cwd: '/repo/server', repo: '/repo', processIdentity: 'Mon Oct 7 12:00:00 2026' };
  assert.equal(nativeBackendVerdict(input).ok, true);
  for (const delta of [{ pids: [] }, { pids: [1] }, { pids: [96833, 96834] }, { executable: '/usr/bin/docker-proxy' },
    { executable: 'server-cli' }, { cwd: '/different/server' }, { processIdentity: null }]) {
    assert.equal(nativeBackendVerdict({ ...input, ...delta }).ok, false);
  }
});

test('controlled fixture serves only its Card, rejects invocation, and restarts at the recorded URL', async t => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'mooc-a2a-card-test-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  let active;
  try {
    active = await startCardFixture(directory);
    const saved = active.descriptor;
    assert.equal(validFixtureIdentity(saved), true);
    const response = await fetch(saved.baseUrl + '/.well-known/agent-card.json');
    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), saved.card);
    assert.equal((await fetch(saved.baseUrl, { method: 'POST', body: '' })).status, 405);
    assert.equal((await fetch(saved.baseUrl + '/other')).status, 404);
    assert.equal(active.observations.filter(item => item.servedCard).length, 1);
    await active.close();
    active = await startCardFixture(directory, JSON.parse(fs.readFileSync(path.join(directory, 'fixture.json'), 'utf8')));
    assert.equal(active.descriptor.baseUrl, saved.baseUrl);
    assert.deepEqual(await (await fetch(saved.baseUrl + '/.well-known/agent-card.json')).json(), saved.card);
  } finally { await active?.close(); }
});


test('A2A ownership pins the first ID and preserves its evidence across normal and recovery reads', () => {
  const confirmations = [];
  const ownership = createA2aOwnership([existing], fixture, row => confirmations.push({ ...row }));
  assert.equal(ownership.id, null);
  assert.equal(ownership.find([existing]), null);
  assert.deepEqual(ownership.find([existing, owned]), owned);
  assert.equal(ownership.id, owned.id);
  assert.deepEqual(ownership.find([existing, { ...owned, enabled: false }]), { ...owned, enabled: false });
  const replacement = { ...owned, id: '33333333-3333-4333-9333-333333333333' };
  for (let attempt = 0; attempt < 2; attempt++) {
    assert.throws(() => ownership.find([existing, replacement]), error =>
      error.code === 'A2A_OWNERSHIP_CONFLICT' && error.candidate.id === replacement.id);
    assert.equal(ownership.id, owned.id);
    assert.deepEqual(confirmations, [owned], 'First proof must never be overwritten by a replacement');
  }
  assert.equal(ownership.find([existing]), null);
  assert.equal(ownership.id, owned.id, 'Disappearance does not release the first identity');
});

test('replacement A2A identity cannot pass a later write or recovery check', () => {
  const replacement = { ...owned, id: '33333333-3333-4333-9333-333333333333' };
  for (const action of ['enabled', 'delete']) {
    assert.equal(allowedConfigWrite(write({ url: `http://127.0.0.1:4317${a2aPath}/${replacement.id}/${action}`,
      payload: action === 'enabled' ? { enabled: false } : null, body: null,
      expected: { ...expected(action), owned: replacement, knownId: owned.id },
    })), false);
  }
  assert.throws(() => a2aRecoveryStatus({ journal: new ConfigWriteJournal(), baseline: [existing], current: [existing, replacement],
    fixture, knownId: owned.id, createAttempted: true, deleteConfirmed: false }), /owned ID changed/);
});
