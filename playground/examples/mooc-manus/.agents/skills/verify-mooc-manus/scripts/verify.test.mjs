#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { recordBuildConsistency, resolveRunDirectory, sourceFingerprint } from './verify.mjs';

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
