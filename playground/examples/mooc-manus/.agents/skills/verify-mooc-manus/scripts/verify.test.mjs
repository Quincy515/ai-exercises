#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { excludeLlmDevStream, forwardLlmRead } from './llm-transport.mjs';
import { applyFinalSafetyGate, settleRouteHandlers, trackRouteHandler } from './route-lifecycle.mjs';
import { llmInputMatches, llmPayload, llmRestoreDecision, llmTarget, readLlmConfig, sameLlmConfig, validLlmPayload } from './llm-policy.mjs';
import { recordBuildConsistency, resolveRunDirectory, sourceFingerprint, viteLaunchCommand } from './verify.mjs';
import { allowedConfigWrite, ConfigWriteJournal, forwardAuthorizedConfigWrite, requireConfigWriteAuthorization, restoreConfigAfterWrites, restoreDecision, withinPostDeadline } from './config-write.mjs';

const temporary = t => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'mooc-verification-test-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  return directory;
};
const write = (root, relative, contents) => {
  const file = path.join(root, relative);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, contents);
};
const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));

for (const file of ['apps/packages/tsconfig.json', 'apps/tanstack-app/tsconfig.json',
  'apps/tanstack-app/tsr.config.json', 'apps/shared/Cargo.toml', 'apps/shared/boltffi.toml']) {
  test(`fingerprint detects edits and deletion of ${file}`, t => {
    const root = temporary(t);
    write(root, file, 'first configuration');
    const before = sourceFingerprint(root);
    write(root, file, 'changed configuration');
    const after = sourceFingerprint(root);
    assert.notEqual(after.digest, before.digest);
    assert.notEqual(after.files[file], before.files[file]);
    fs.unlinkSync(path.join(root, file));
    const deleted = sourceFingerprint(root);
    assert.notEqual(deleted.digest, after.digest);
    assert.equal(deleted.files[file], undefined);
  });
}

test('changed build invalidates summary and feature evidence consistently', t => {
  const root = temporary(t);
  const report = { features: [
    { id: 'sessions', status: 'passed', checks: ['navigation'] },
    { id: 'settings', status: 'blocked', reason: 'backend unavailable' },
    { id: 'files', status: 'failed', reason: 'original assertion failure' },
  ] };
  write(root, 'results.json', JSON.stringify(report));
  for (const feature of report.features) write(root, `${feature.id}/result.json`, JSON.stringify(feature));
  const final = { digest: 'changed-build', files: {} };
  const updated = recordBuildConsistency(root, 'original-build', final, report);
  assert.equal(updated, report, 'caller must observe invalidated statuses for its exit code');
  assert.equal(updated.buildUnchanged, false);
  assert.match(updated.invalidated, /Working tree changed/);
  assert.deepEqual(updated.features.map(feature => feature.status), ['failed', 'blocked', 'failed']);
  assert.deepEqual(read(path.join(root, 'results.json')), updated);
  for (const feature of updated.features) {
    assert.equal(feature.buildUnchanged, false);
    assert.equal(feature.invalidated, updated.invalidated);
    assert.deepEqual(read(path.join(root, feature.id, 'result.json')), feature);
  }
  assert.equal(read(path.join(root, 'working-tree-after.json')).unchanged, false);
});

test('unchanged build preserves feature outcome and records consistency', t => {
  const root = temporary(t);
  const report = { features: [{ id: 'sessions', status: 'passed', checks: ['navigation'] }] };
  write(root, 'results.json', JSON.stringify(report));
  write(root, 'sessions/result.json', JSON.stringify(report.features[0]));
  const updated = recordBuildConsistency(root, 'same-build', { digest: 'same-build', files: {} });
  assert.equal(updated.buildUnchanged, true);
  assert.equal(updated.invalidated, undefined);
  assert.equal(updated.features[0].status, 'passed');
  assert.equal(updated.features[0].buildUnchanged, true);
  assert.deepEqual(read(path.join(root, 'sessions/result.json')), updated.features[0]);
});

test('missing feature artifacts remain absent after infrastructure failure', t => {
  const root = temporary(t);
  const report = { features: [{ id: 'sessions', status: 'unverified' }] };
  write(root, 'results.json', JSON.stringify(report));
  recordBuildConsistency(root, 'before', { digest: 'after', files: {} });
  assert.equal(fs.existsSync(path.join(root, 'sessions')), false);
  assert.equal(read(path.join(root, 'results.json')).features[0].status, 'unverified');
});

test('run paths accept only direct children of the evidence root', t => {
  const root = temporary(t);
  assert.equal(resolveRunDirectory(path.join(root, 'run-a'), root), path.join(root, 'run-a'));
  for (const run of [root, path.join(root, 'group/run-a'), path.join(root, '../elsewhere')]) {
    assert.throws(() => resolveRunDirectory(run, root), /direct child/);
  }
});

test('launch and state-reading CLI reject nested run paths before side effects', () => {
  const script = fileURLToPath(new URL('./verify.mjs', import.meta.url));
  const repo = path.resolve(path.dirname(script), '../../../..');
  const nested = path.join(repo, 'output/playwright/verify-mooc-manus/group/run-a');
  for (const verb of ['launch', 'doctor', 'drive', 'cleanup']) {
    const result = spawnSync(process.execPath, [script, verb, '--run', nested], { encoding: 'utf8' });
    assert.equal(result.status, 1, verb);
    assert.match(result.stderr, /direct child/, verb);
  }
});

test('settings-save requires an explicit boolean grant and keeps default flows read-only', () => {
  for (const permission of [undefined, false, 'true', 1, 'false']) {
    assert.throws(() => requireConfigWriteAuthorization(['settings-save'], permission), /--allow-config-write true/);
  }
  assert.doesNotThrow(() => requireConfigWriteAuthorization(['settings-save'], true));
  assert.doesNotThrow(() => requireConfigWriteAuthorization(['settings', 'sessions', 'files']));
});

test('CLI refuses unauthorized write flow before launch or instance lookup', () => {
  const script = fileURLToPath(new URL('./verify.mjs', import.meta.url));
  const repo = path.resolve(path.dirname(script), '../../../..');
  const run = path.join(repo, 'output/playwright/verify-mooc-manus/unauthorized-selftest');
  assert.equal(fs.existsSync(run), false);
  for (const verb of ['launch', 'run', 'drive']) {
    for (const permission of [[], ['--allow-config-write', 'false'], ['--allow-config-write', 'TRUE']]) {
      const result = spawnSync(process.execPath, [script, verb, '--run', run, '--features', 'settings-save', ...permission], { encoding: 'utf8' });
      assert.equal(result.status, 1);
      assert.match(result.stderr, /settings-save requires explicit --allow-config-write true/);
      assert.equal(fs.existsSync(run), false, 'denial must not create state or launch an instance');
    }
  }
});

const originalConfig = { max_iterations: 100, max_retries: 3, max_search_results: 10 };
const writePolicy = overrides => ({ feature: 'settings-save', allowConfigWrite: true, baseUrl: 'http://127.0.0.1:4317',
  url: 'http://127.0.0.1:4317/api/app_configs/agent', method: 'POST', payload: originalConfig, expected: originalConfig, ...overrides });

test('write guard permits only the opted-in flow and exact same-origin POST endpoint', () => {
  assert.equal(allowedConfigWrite(writePolicy()), true);
  for (const overrides of [
    { feature: 'settings' }, { feature: 'sessions' }, { feature: 'files' }, { allowConfigWrite: false },
    { method: 'PUT' }, { method: 'DELETE' }, { method: 'PATCH' },
    { url: 'http://localhost:4317/api/app_configs/agent' },
    { url: 'http://127.0.0.1:4318/api/app_configs/agent' },
    { url: 'https://127.0.0.1:4317/api/app_configs/agent' },
    { url: 'http://127.0.0.1:4317/api/app_configs/agent?test=1' },
    { url: 'http://127.0.0.1:4317/api/app_configs/agent/' },
    { url: 'http://127.0.0.1:4317/api/app_configs/llm' },
    { expected: null },
  ]) assert.equal(allowedConfigWrite(writePolicy(overrides)), false, JSON.stringify(overrides));
});

test('write guard accepts only exact expected three-field integer payload', () => {
  for (const payload of [null, [], 'text', {}, { ...originalConfig, extra: true },
    { max_iterations: 100, max_retries: 3 }, { ...originalConfig, max_iterations: '100' },
    { ...originalConfig, max_iterations: 101 }, { ...originalConfig, max_retries: 4 },
    { ...originalConfig, max_search_results: 11 },
  ]) assert.equal(allowedConfigWrite(writePolicy({ payload })), false);
});

test('write guard rejects expected values outside backend bounds', () => {
  for (const [key, invalid] of [['max_iterations', 0], ['max_iterations', 1000], ['max_iterations', 1.5],
    ['max_retries', 1], ['max_retries', 10], ['max_search_results', 1], ['max_search_results', 30]]) {
    const payload = { ...originalConfig, [key]: invalid };
    assert.equal(allowedConfigWrite(writePolicy({ payload, expected: payload })), false);
  }
  for (const payload of [{ max_iterations: 1, max_retries: 2, max_search_results: 2 },
    { max_iterations: 999, max_retries: 9, max_search_results: 29 }]) {
    assert.equal(allowedConfigWrite(writePolicy({ payload, expected: payload })), true);
  }
});

test('cleanup restores only the test target and preserves concurrent third-party changes', () => {
  const target = { ...originalConfig, max_iterations: 101 };
  assert.equal(restoreDecision({ ...originalConfig }, originalConfig, target), 'already-restored');
  assert.equal(restoreDecision({ ...target }, originalConfig, target), 'restore');
  for (const current of [{ ...target, max_retries: 4 }, { ...originalConfig, max_iterations: 200 }, null]) {
    assert.throws(() => restoreDecision(current, originalConfig, target), /third-party value/);
  }
});

test('POST timeout with a late server commit never reports restoration from an immediate GET', async () => {
  const snapshots = [];
  const journal = new ConfigWriteJournal(entries => snapshots.push(structuredClone(entries)));
  const target = { ...originalConfig, max_iterations: 101 };
  const entry = journal.begin({ url: writePolicy().url, payload: target });
  let current = { ...originalConfig };
  let release;
  const delayed = new Promise(resolve => { release = resolve; });
  const operation = (async () => {
    await delayed;
    current = { ...target }; // Server commits after the browser's deadline.
    journal.complete(entry, 200);
  })();
  await assert.rejects(withinPostDeadline(() => operation, journal, 5), /deadline exceeded/);
  assert.equal(entry.outcome, 'outcome-unknown');
  assert.equal(snapshots.at(-1)[0].outcome, 'outcome-unknown');
  let gets = 0, posts = 0;
  const progress = {};
  const restore = () => restoreConfigAfterWrites({ journal, original: originalConfig, target, progress,
    read: async () => { gets++; return { ...current }; }, write: async () => { posts++; } });
  await assert.rejects(restore(), /outcome-unknown/);
  assert.equal(gets, 0, 'Immediate GET could show the old value while POST is still executing');
  assert.equal(posts, 0);
  assert.equal(progress.restored, undefined);
  release();
  await operation;
  assert.deepEqual(current, target);
  assert.equal(entry.outcome, 'outcome-unknown', 'Late response is evidence, not a retroactive successful run');
  assert.equal(entry.lateResponseReceived, true);
  await assert.rejects(restore(), /outcome-unknown/);
  assert.equal(progress.restored, undefined);
  assert.equal(posts, 0);
});

test('transport failure preserves the authorized POST and blocks automatic compensation', async () => {
  const journal = new ConfigWriteJournal();
  const target = { ...originalConfig, max_iterations: 101 };
  const entry = journal.begin({ url: writePolicy().url, payload: target });
  await assert.rejects(withinPostDeadline(async () => { throw new Error('net::ERR_CONNECTION_RESET'); }, journal), /CONNECTION_RESET/);
  assert.equal(entry.outcome, 'outcome-unknown');
  assert.deepEqual(entry.payload, target);
  let observed = false;
  const progress = {};
  await assert.rejects(restoreConfigAfterWrites({ journal, original: originalConfig, target, progress,
    read: async () => { observed = true; return originalConfig; },
    write: async () => { observed = true; } }), /outcome-unknown/);
  assert.equal(observed, false);
  assert.equal(progress.decision, 'outcome-unknown');
  assert.equal(progress.restored, undefined);
});

test('response headers alone leave POST outcome unknown at cleanup', () => {
  const journal = new ConfigWriteJournal();
  const entry = journal.begin({ url: writePolicy().url, payload: originalConfig });
  journal.headers(entry, 200);
  journal.finalize();
  assert.equal(entry.status, 200);
  assert.equal(entry.outcome, 'outcome-unknown');
});

test('completed POST response preserves ordinary compensating restore and final GET verification', async () => {
  const journal = new ConfigWriteJournal();
  const target = { ...originalConfig, max_iterations: 101 };
  const entry = journal.begin({ url: writePolicy().url, payload: target });
  journal.complete(entry, 200);
  journal.finalize();
  let current = { ...target }, gets = 0;
  const writes = [];
  const progress = await restoreConfigAfterWrites({ journal, original: originalConfig, target,
    read: async () => { gets++; return { ...current }; },
    write: async values => { writes.push(values); current = { ...values }; } });
  assert.equal(entry.outcome, 'response-received');
  assert.deepEqual(writes, [originalConfig]);
  assert.equal(gets, 2);
  assert.equal(progress.restored, true);
  assert.equal(progress.compensatingWrite, true);
  assert.deepEqual(progress.after, originalConfig);
});

test('compensation transport failure keeps restoration unproven', async () => {
  const journal = new ConfigWriteJournal();
  const target = { ...originalConfig, max_iterations: 101 };
  journal.complete(journal.begin({ url: writePolicy().url, payload: target }), 200);
  const progress = {};
  let gets = 0;
  await assert.rejects(restoreConfigAfterWrites({ journal, original: originalConfig, target, progress,
    read: async () => { gets++; return target; }, write: async values => {
      journal.begin({ url: writePolicy().url, payload: values, source: 'compensation' });
      await withinPostDeadline(async () => { throw new Error('restore timeout'); }, journal);
    } }), /restore timeout/);
  assert.equal(gets, 1);
  assert.equal(progress.restored, undefined);
  assert.equal(journal.entries[1].outcome, 'outcome-unknown');
  assert.equal(journal.entries[1].source, 'compensation');
});

test('launcher directly uses local Vite JavaScript with Node and the app working directory', t => {
  const root = path.join(temporary(t), 'checkout with spaces');
  write(root, 'apps/tanstack-app/package.json', JSON.stringify({ scripts: { dev: 'vite dev --port 3000' } }));
  write(root, 'apps/node_modules/vite/package.json', JSON.stringify({ name: 'vite', version: '8.0.0' }));
  write(root, 'apps/node_modules/vite/bin/vite.js', '// Existing local Vite CLI fixture');
  const launch = viteLaunchCommand(root, 4321);
  assert.equal(launch.bin, process.execPath);
  assert.equal(launch.cwd, path.join(root, 'apps/tanstack-app'));
  assert.deepEqual(launch.args, [fs.realpathSync(path.join(root, 'apps/node_modules/vite/bin/vite.js')),
    'dev', '--host', '127.0.0.1', '--port', '4321', '--strictPort']);
  assert.equal(launch.devScript, 'vite dev --port 3000');
});

test('launcher rejects changed or shell-composed dev scripts before resolving tools', t => {
  const root = temporary(t);
  for (const dev of ['vite', 'vite dev --port 3001', 'vite dev --port 3000 && other-command',
    'VITE_API_BASE_URL=https://example.test vite dev --port 3000', 'pnpm exec vite dev --port 3000', undefined]) {
    write(root, 'apps/tanstack-app/package.json', JSON.stringify({ scripts: { dev } }));
    assert.throws(() => viteLaunchCommand(root, 4317), /Unsupported tanstack-app scripts.dev/);
  }
});

test('launcher reports an absent local Vite CLI without installing a dependency', t => {
  const root = temporary(t);
  write(root, 'apps/tanstack-app/package.json', JSON.stringify({ scripts: { dev: 'vite dev --port 3000' } }));
  write(root, 'apps/node_modules/vite/package.json', JSON.stringify({ name: 'vite', version: '8.0.0' }));
  assert.throws(() => viteLaunchCommand(root, 4317), /Local Vite JavaScript CLI unavailable/);
  assert.equal(fs.existsSync(path.join(root, 'apps/node_modules/vite/bin/vite.js')), false);
});

test('authorized write forwards the same real response with redirects and retries disabled', async () => {
  const journal = new ConfigWriteJournal();
  const entry = journal.begin({ url: writePolicy().url, payload: originalConfig });
  const response = { status: () => 200 };
  const calls = [];
  await forwardAuthorizedConfigWrite({
    fetch: async options => { calls.push(['fetch', options]); return response; },
    fulfill: async options => { calls.push(['fulfill', options]); },
    abort: async reason => { calls.push(['abort', reason]); },
  }, journal, entry);
  assert.deepEqual(calls, [['fetch', { maxRedirects: 0, maxRetries: 0, timeout: 10000 }], ['fulfill', { response }]]);
  assert.equal(entry.outcome, 'response-received');
  assert.equal(entry.status, 200);
});

for (const status of [307, 308]) {
  test(`authorized write rejects ${status} before the browser receives the redirect`, async () => {
    const journal = new ConfigWriteJournal();
    const entry = journal.begin({ url: writePolicy().url, payload: originalConfig });
    const calls = [];
    await forwardAuthorizedConfigWrite({
      fetch: async options => { calls.push(['fetch', options]); return { status: () => status }; },
      fulfill: async () => { calls.push(['fulfill']); },
      abort: async reason => { calls.push(['abort', reason]); },
    }, journal, entry);
    assert.deepEqual(calls, [['fetch', { maxRedirects: 0, maxRetries: 0, timeout: 10000 }], ['abort', 'blockedbyclient']]);
    assert.equal(entry.outcome, 'outcome-unknown');
    assert.equal(entry.status, status);
  });
}

test('authorized write transport error records unknown and aborts the browser request', async () => {
  const journal = new ConfigWriteJournal();
  const entry = journal.begin({ url: writePolicy().url, payload: originalConfig });
  let aborts = 0;
  let fulfills = 0;
  await forwardAuthorizedConfigWrite({
    fetch: async () => { throw new Error('socket hang up'); },
    fulfill: async () => { fulfills++; },
    abort: async () => { aborts++; },
  }, journal, entry);
  assert.equal(aborts, 1);
  assert.equal(fulfills, 0);
  assert.equal(entry.outcome, 'outcome-unknown');
  assert.match(entry.reason, /socket hang up/);
});

const llmOriginal = { base_url: null, model_name: null, temperature: null, max_tokens: null, api_key_configured: true };
const llmWritePolicy = overrides => ({ feature: 'llm-save', allowConfigWrite: true, baseUrl: 'http://127.0.0.1:4317',
  url: 'http://127.0.0.1:4317/api/app_configs/llm', method: 'POST', payload: llmPayload(llmOriginal), expected: llmPayload(llmOriginal), ...overrides });

test('llm-save refuses missing explicit authorization before any instance lookup', () => {
  assert.throws(() => requireConfigWriteAuthorization(['llm-save'], false), /llm-save requires explicit/);
  const script = fileURLToPath(new URL('./verify.mjs', import.meta.url));
  for (const verb of ['launch', 'run', 'drive']) {
    const result = spawnSync(process.execPath, [script, verb, '--features', 'llm-save'], { encoding: 'utf8' });
    assert.equal(result.status, 1);
    assert.match(result.stderr, /llm-save requires explicit --allow-config-write true/);
  }
});

test('LLM response evidence is a five-field whitelist and drops secrets and unknown nested data', () => {
  const decoded = readLlmConfig(JSON.stringify({ ...llmOriginal, api_key: 'NEVER-LOG-THIS-KEY', nested: { secret: 'PRIVATE' } }));
  assert.deepEqual(decoded, llmOriginal);
  assert.equal(JSON.stringify(decoded).includes('NEVER-LOG'), false);
  assert.deepEqual(Object.keys(decoded).sort(), ['api_key_configured', 'base_url', 'max_tokens', 'model_name', 'temperature']);
  assert.throws(() => readLlmConfig('NEVER-LOG-THIS-KEY'), error => {
    assert.equal(error.message.includes('NEVER-LOG'), false); return true;
  });
});

test('LLM unsafe max_tokens blocks before any rounded value can enter evidence', () => {
  const text = '{"base_url":null,"model_name":null,"temperature":null,"max_tokens":9223372036854775807,"api_key_configured":true,"api_key":"NEVER-LOG-THIS-KEY"}';
  assert.throws(() => readLlmConfig(text), error => {
    assert.equal(error.code, 'LLM_UNSAFE_MAX_TOKENS');
    assert.equal(error.message.includes('922337'), false);
    assert.equal(error.message.includes('NEVER-LOG'), false);
    return true;
  });
});

test('LLM write guard rejects every api_key property, including empty and null', () => {
  assert.equal(allowedConfigWrite(llmWritePolicy()), true);
  for (const api_key of ['', null, 'NEVER-SEND-THIS-KEY']) {
    const payload = { ...llmPayload(llmOriginal), api_key };
    assert.equal(allowedConfigWrite(llmWritePolicy({ payload })), false);
    assert.equal(allowedConfigWrite(llmWritePolicy({ payload, expected: payload })), false);
  }
  assert.equal(allowedConfigWrite(llmWritePolicy({ payload: llmOriginal })), false, 'Response key status does not belong in a write');
});

test('LLM write guard permits only opted-in same-origin exact endpoint and expected public fields', () => {
  for (const overrides of [{ feature: 'llm' }, { feature: 'settings-save' }, { allowConfigWrite: false },
    { method: 'PUT' }, { method: 'DELETE' }, { url: 'https://model.example/api/app_configs/llm' },
    { url: 'http://127.0.0.1:4317/api/app_configs/agent' }, { url: 'http://127.0.0.1:4317/api/app_configs/llm?key=hidden' },
    { payload: { ...llmPayload(llmOriginal), max_tokens: 1 } },
  ]) assert.equal(allowedConfigWrite(llmWritePolicy(overrides)), false);
});

test('LLM public payload validates nullable fields and integer and temperature boundaries', () => {
  for (const values of [llmPayload(llmOriginal), { base_url: 'https://example.test/v1', model_name: 'model', temperature: -2, max_tokens: 0 },
    { base_url: null, model_name: null, temperature: 2, max_tokens: Number.MAX_SAFE_INTEGER }]) assert.equal(validLlmPayload(values), true);
  for (const delta of [{ max_tokens: -1 }, { max_tokens: 1.5 }, { max_tokens: Number.MAX_SAFE_INTEGER + 1 },
    { temperature: 3 }, { temperature: Infinity }, { temperature: '1' }, { model_name: 7 }]) {
    assert.equal(validLlmPayload({ ...llmPayload(llmOriginal), ...delta }), false);
  }
});

test('LLM target changes only max_tokens and keeps a nullable original recoverable', () => {
  assert.deepEqual(llmTarget(llmOriginal), { ...llmOriginal, max_tokens: 1 });
  assert.equal(llmOriginal.max_tokens, null);
  assert.equal(llmTarget({ ...llmOriginal, max_tokens: 0 }).max_tokens, 1);
  assert.equal(llmTarget({ ...llmOriginal, max_tokens: Number.MAX_SAFE_INTEGER }).max_tokens, Number.MAX_SAFE_INTEGER - 1);
});

test('LLM compensation restores explicit nulls while omitting api_key entirely', async () => {
  const journal = new ConfigWriteJournal();
  const target = llmTarget(llmOriginal);
  journal.complete(journal.begin({ url: llmWritePolicy().url, payload: llmPayload(target) }), 200);
  let current = { ...target };
  const writes = [];
  const progress = await restoreConfigAfterWrites({ journal, original: llmOriginal, target,
    equals: sameLlmConfig, decide: llmRestoreDecision,
    read: async () => current, write: async values => {
      const payload = llmPayload(values);
      writes.push(payload); current = { ...payload, api_key_configured: true };
    } });
  assert.deepEqual(writes, [{ base_url: null, model_name: null, temperature: null, max_tokens: null }]);
  assert.equal(progress.restored, true);
  assert.equal(progress.after.api_key_configured, true);
});

test('LLM third-party field or key status changes prevent automatic restore', async () => {
  const target = llmTarget(llmOriginal);
  for (const current of [{ ...target, model_name: 'someone-else' }, { ...target, api_key_configured: false }]) {
    const journal = new ConfigWriteJournal();
    journal.complete(journal.begin({ url: llmWritePolicy().url, payload: llmPayload(target) }), 200);
    let writes = 0;
    await assert.rejects(restoreConfigAfterWrites({ journal, original: llmOriginal, target,
      equals: sameLlmConfig, decide: llmRestoreDecision, read: async () => current, write: async () => { writes++; } }), /cleanup conflict/);
    assert.equal(writes, 0);
  }
});

test('unknown LLM POST result prevents even an immediate GET from claiming restoration', async () => {
  const target = llmTarget(llmOriginal), journal = new ConfigWriteJournal();
  const entry = journal.begin({ url: llmWritePolicy().url, payload: llmPayload(target) });
  journal.unknown(entry, 'POST timeout');
  let io = 0;
  await assert.rejects(restoreConfigAfterWrites({ journal, original: llmOriginal, target,
    equals: sameLlmConfig, decide: llmRestoreDecision, read: async () => { io++; return llmOriginal; }, write: async () => { io++; } }), /outcome-unknown/);
  assert.equal(io, 0);
});

test('LLM temperature accepts f32-equivalent scientific and decimal renderings only', () => {
  for (const [actual, expected] of [['0.0000001', 1e-7], ['1e-7', 1e-7], ['-0.0000001', -1e-7],
    ['-1e-7', -1e-7], ['0', 0], ['-0', 0], ['0.7', 0.7], ['0.699999988079071', 0.7]]) {
    assert.equal(llmInputMatches('temperature', actual, expected), true, `${actual} / ${expected}`);
  }
  for (const [actual, expected] of [['', 0], [' ', 0], ['NaN', 0], ['Infinity', 0], ['1e40', 1e40],
    ['0.0000002', 1e-7], ['0.8', 0.7], ['0.7', NaN], ['0.7', Infinity]]) {
    assert.equal(llmInputMatches('temperature', actual, expected), false, `${actual} / ${expected}`);
  }
});

test('LLM nullable, string and max_tokens input matching preserves exact text and integer precision', () => {
  for (const field of ['base_url', 'model_name', 'temperature', 'max_tokens']) {
    assert.equal(llmInputMatches(field, '', null), true);
    assert.equal(llmInputMatches(field, ' ', null), false);
    assert.equal(llmInputMatches(field, '0', null), false);
  }
  assert.equal(llmInputMatches('model_name', 'model-name', 'model-name'), true);
  assert.equal(llmInputMatches('model_name', 'model-name ', 'model-name'), false);
  assert.equal(llmInputMatches('max_tokens', '9007199254740991', Number.MAX_SAFE_INTEGER), true);
  assert.equal(llmInputMatches('max_tokens', '9007199254740990', Number.MAX_SAFE_INTEGER), false);
  assert.equal(llmInputMatches('max_tokens', '9.007199254740991e15', Number.MAX_SAFE_INTEGER), false);
  assert.equal(llmInputMatches('max_tokens', '01', 1), false);
  assert.equal(llmInputMatches('api_key', '', null), false);
});

for (const method of ['GET', 'HEAD']) {
  test(`LLM ${method} forwards the same real 200 response with redirects and retries disabled`, async () => {
    const response = { status: () => 200 }, calls = [], blocked = [];
    await forwardLlmRead({ request: () => ({ method: () => method, url: () => 'http://127.0.0.1:4317/api/app_configs/llm' }),
      fetch: async options => { calls.push(['fetch', options]); return response; },
      fulfill: async options => { calls.push(['fulfill', options]); },
      abort: async reason => { calls.push(['abort', reason]); },
    }, blocked);
    assert.deepEqual(calls, [['fetch', { maxRedirects: 0, maxRetries: 0, timeout: 10000 }], ['fulfill', { response }]]);
    assert.deepEqual(blocked, []);
  });
  for (const status of [302, 307]) {
    test(`LLM ${method} rejects ${status} and records only a sanitized source URL`, async () => {
      const blocked = [], calls = [];
      await forwardLlmRead({ request: () => ({ method: () => method, url: () => 'http://127.0.0.1:4317/asset.js?secret=NEVER-LOG' }),
        fetch: async options => { calls.push(options); return { status: () => status }; },
        fulfill: async () => { calls.push('fulfill'); },
        abort: async reason => { calls.push(reason); },
      }, blocked);
      assert.deepEqual(calls, [{ maxRedirects: 0, maxRetries: 0, timeout: 10000 }, 'blockedbyclient']);
      assert.deepEqual(blocked, [{ method, url: 'http://127.0.0.1:4317/asset.js', reason: 'LLM read redirect blocked', status }]);
      assert.equal(JSON.stringify(blocked).includes('NEVER-LOG'), false);
    });
  }
}

test('LLM read transport errors use a fixed safe reason', async () => {
  const blocked = [];
  let aborted = false, fulfilled = false;
  await forwardLlmRead({ request: () => ({ method: () => 'GET', url: () => 'http://127.0.0.1:4317/api/app_configs/llm?token=NEVER-LOG' }),
    fetch: async () => { throw new Error('network detail contains NEVER-LOG'); },
    fulfill: async () => { fulfilled = true; },
    abort: async () => { aborted = true; },
  }, blocked);
  assert.equal(aborted, true);
  assert.equal(fulfilled, false);
  assert.deepEqual(blocked, [{ method: 'GET', url: 'http://127.0.0.1:4317/api/app_configs/llm', reason: 'LLM read transport failed' }]);
});

test('LLM read helper rejects mutating methods before network access', async () => {
  const blocked = [];
  let requests = 0;
  await forwardLlmRead({ request: () => ({ method: () => 'POST', url: () => 'http://127.0.0.1:4317/api/app_configs/llm' }),
    fetch: async () => { requests++; return { status: () => 200 }; }, fulfill: async () => {}, abort: async () => {},
  }, blocked);
  assert.equal(requests, 0);
  assert.equal(blocked[0].reason, 'LLM read transport accepts only GET/HEAD');
});

test('LLM excludes only the local devtools GET stream without upstream access', async () => {
  const ignored = [], calls = [];
  const matched = await excludeLlmDevStream({
    request: () => ({ method: () => 'GET', url: () => 'http://127.0.0.1:4317/__tsd/console-pipe/sse?secret=omit' }),
    abort: async reason => { calls.push(['abort', reason]); },
    fetch: async () => { calls.push('fetch'); }, continue: async () => { calls.push('continue'); },
  }, 'http://127.0.0.1:4317', ignored);
  assert.equal(matched, true);
  assert.deepEqual(calls, [['abort', 'blockedbyclient']]);
  assert.deepEqual(ignored, [{ method: 'GET', url: 'http://127.0.0.1:4317/__tsd/console-pipe/sse', reason: 'devtools stream excluded' }]);
});

test('devtools exclusion never broadens to other origins, methods or paths', async () => {
  for (const [method, url] of [['HEAD', 'http://127.0.0.1:4317/__tsd/console-pipe/sse'],
    ['POST', 'http://127.0.0.1:4317/__tsd/console-pipe/sse'], ['GET', 'http://elsewhere.test/__tsd/console-pipe/sse'],
    ['GET', 'http://127.0.0.1:4317/__tsd/console-pipe/sse/'], ['GET', 'http://127.0.0.1:4317/api/app_configs/llm']]) {
    const ignored = []; let aborted = false;
    assert.equal(await excludeLlmDevStream({ request: () => ({ method: () => method, url: () => url }),
      abort: async () => { aborted = true; } }, 'http://127.0.0.1:4317', ignored), false);
    assert.equal(aborted, false); assert.deepEqual(ignored, []);
  }
});

const cleanEvidence = () => ({ status: 'passed', blockedWrites: [], pageErrors: [], postOutcomes: [] });

test('final safety gate waits for a late blocked route before persisting a passed result', async () => {
  const pending = new Set(), result = cleanEvidence();
  let release;
  const afterClose = new Promise(resolve => { release = resolve; });
  trackRouteHandler(pending, async () => {
    await afterClose;
    result.blockedWrites.push({ method: 'GET', reason: 'LLM read transport failed' });
  }, error => result.pageErrors.push(error.message));
  assert.equal(result.status, 'passed');
  assert.deepEqual(result.blockedWrites, [], 'The old pre-close check would incorrectly pass');
  const settled = settleRouteHandlers(pending);
  release();
  await settled;
  applyFinalSafetyGate(result);
  const persisted = JSON.parse(JSON.stringify(result));
  assert.equal(pending.size, 0);
  assert.equal(persisted.status, 'failed');
  assert.equal(persisted.blockedWrites.length, 1);
  assert.equal(persisted.safetyGate.checkedAfterContextClose, true);
});

test('late route exceptions and unknown POSTs fail the final gate while ignored devtools remain informational', async () => {
  const pending = new Set(), result = cleanEvidence();
  trackRouteHandler(pending, async () => { throw new Error('route stopped during close'); }, error => result.pageErrors.push(error.message));
  await settleRouteHandlers(pending);
  applyFinalSafetyGate(result);
  assert.equal(result.status, 'failed');
  const unknown = cleanEvidence();
  unknown.postOutcomes.push({ outcome: 'outcome-unknown' });
  unknown.configCleanup = { restored: true };
  applyFinalSafetyGate(unknown);
  assert.equal(unknown.status, 'failed');
  assert.equal(unknown.configCleanup.restored, undefined);
  const ignored = cleanEvidence();
  ignored.ignoredDevRequests = [{ reason: 'devtools stream excluded' }];
  applyFinalSafetyGate(ignored);
  assert.equal(ignored.status, 'passed');
});
