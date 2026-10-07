#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import net from 'node:net';
import { spawn, spawnSync } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { driveFlows, driverInfo } from './flows.mjs';
import { requireConfigWriteAuthorization } from './config-write.mjs';
import { llmPath, readLlmConfig } from './llm-policy.mjs';
import { a2aPath, readA2aList } from './a2a-policy.mjs';
import { inspectNativeBackend } from './a2a-card-fixture.mjs';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../..');
const evidenceRoot = path.join(repo, 'output/playwright/verify-mooc-manus');
const backend = 'http://localhost:5150';
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const hash = value => createHash('sha256').update(value).digest('hex');
const json = (file, value) => fs.writeFileSync(file, JSON.stringify(value, null, 2) + '\n');
const command = (bin, args, cwd = repo) => {
  const result = spawnSync(bin, args, { cwd, encoding: 'utf8', timeout: 15000, env: { ...process.env, GIT_OPTIONAL_LOCKS: '0' } });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${bin} ${args.join(' ')}: ${result.stderr.trim() || result.status}`);
  return result.stdout.trim();
};

export function sourceFingerprint(root = repo) {
  const files = {};
  function visit(relative) {
    const absolute = path.join(root, relative);
    if (!fs.existsSync(absolute)) return;
    if (fs.statSync(absolute).isDirectory()) {
      for (const item of fs.readdirSync(absolute).sort()) {
        if (!['node_modules', 'dist', 'target', '.git'].includes(item)) visit(path.join(relative, item));
      }
    } else files[relative] = hash(fs.readFileSync(absolute));
  }
  for (const directory of ['apps/shared/src', 'apps/packages/src', 'apps/tanstack-app/src',
    'apps/tanstack-app/public', '.agents/skills/verify-mooc-manus', 'apps/generated/pkg', 'apps/generated/types/dist']) visit(directory);
  for (const file of ['apps/Cargo.toml', 'apps/Cargo.lock', 'apps/Justfile', 'apps/package.json',
    'apps/pnpm-lock.yaml', 'apps/pnpm-workspace.yaml', 'apps/vite.api.mts',
    'apps/shared/Cargo.toml', 'apps/shared/boltffi.toml',
    'apps/packages/package.json', 'apps/packages/tsconfig.json',
    'apps/tanstack-app/package.json', 'apps/tanstack-app/vite.config.ts',
    'apps/tanstack-app/tsconfig.json', 'apps/tanstack-app/tsr.config.json',
    ...['.env', '.env.local', '.env.development', '.env.development.local'].map(f => `apps/tanstack-app/${f}`)]) visit(file);
  return { digest: hash(JSON.stringify(files)), files };
}

function snapshot() {
  return { head: command('git', ['rev-parse', 'HEAD']),
    status: command('git', ['status', '--short', '--untracked-files=all', '--', '.']),
    ...sourceFingerprint() };
}

function processIdentity(pid) {
  const result = spawnSync('ps', ['-p', String(pid), '-o', 'lstart='], { encoding: 'utf8' });
  if (result.error) throw result.error;
  if (result.status !== 0 && !result.stdout?.trim()) return null;
  return result.stdout.trim().replace(/\s+/g, ' ') || null;
}

function groupMembers(pgid) {
  return command('ps', ['-axo', 'pid=,pgid=,stat=,lstart=']).split('\n').map(line => {
    const [pid, group, stat, ...birth] = line.trim().split(/\s+/);
    return { pid: Number(pid), pgid: Number(group), stat, birth: birth.join(' ') };
  }).filter(p => p.pgid === pgid && !p.stat.startsWith('Z'));
}

function listeners(port) {
  const result = spawnSync('lsof', ['-nP', '-t', `-iTCP:${port}`, '-sTCP:LISTEN'], { encoding: 'utf8' });
  if (result.error) throw result.error;
  if (result.status !== 0 && result.status !== 1) throw new Error(result.stderr);
  return [...new Set(result.stdout.trim().split(/\s+/).filter(Boolean).map(Number))];
}

export function resolveRunDirectory(run, root = evidenceRoot) {
  const resolved = path.resolve(run);
  if (path.dirname(resolved) !== path.resolve(root)) {
    throw new Error('Run must be a direct child of output/playwright/verify-mooc-manus/');
  }
  return resolved;
}

function readState(run) {
  const resolved = resolveRunDirectory(run);
  const state = JSON.parse(fs.readFileSync(path.join(resolved, 'instance.json'), 'utf8'));
  if (state.repo !== repo || state.run !== resolved || state.schema !== 'verify-mooc-manus/v1') throw new Error('Instance identity mismatch');
  if (!Number.isInteger(state.pid) || state.pid <= 1 || !Number.isInteger(state.port)) throw new Error('Invalid process identity');
  return state;
}

function owned(state) {
  const identity = state.pid && processIdentity(state.pid);
  if (!identity || identity !== state.processIdentity) return false;
  return listeners(state.port).every(pid =>
    Number(command('ps', ['-p', String(pid), '-o', 'pgid='])) === state.pid);
}

async function get(url) {
  try {
    const response = await fetch(url, { signal: AbortSignal.timeout(4000), redirect: 'error' });
    const body = await response.text();
    return { available: response.ok, url, status: response.status, body };
  } catch (error) { return { available: false, url, error: error.message }; }
}

async function doctor(state, features = state.features ?? ['settings', 'sessions', 'files']) {
  const ownership = !state.stoppedAt && owned(state) && listeners(state.port).length > 0;
  const buildUnchanged = snapshot().digest === state.source.digest;
  const page = ownership ? await get(state.url) : { available: false, error: 'Owned listener absent' };
  const api = await get(`${backend}/api/app_configs/agent`);
  let config;
  try {
    config = JSON.parse(api.body);
    api.contract = ['max_iterations', 'max_retries', 'max_search_results'].every(k => Number.isSafeInteger(config[k]));
  } catch { api.contract = false; }
  // Only the three public values are recorded, never arbitrary backend error bodies.
  delete api.body;
  if (api.contract) api.config = Object.fromEntries(['max_iterations', 'max_retries', 'max_search_results'].map(k => [k, config[k]]));
  let llmBackend;
  if (features.some(feature => ['llm', 'llm-save'].includes(feature))) {
    llmBackend = await get(backend + llmPath);
    try { llmBackend.config = readLlmConfig(llmBackend.body); llmBackend.contract = true; }
    catch (error) { llmBackend.contract = false; llmBackend.reason = error.code === 'LLM_UNSAFE_MAX_TOKENS'
      ? error.message : 'LLM response does not satisfy the public configuration contract'; }
    delete llmBackend.body;
  }
  let a2aBackend, a2aLocalBackend;
  if (features.some(feature => ['a2a', 'a2a-write'].includes(feature))) {
    a2aBackend = await get(backend + a2aPath);
    try { a2aBackend.a2a_servers = readA2aList(a2aBackend.body); a2aBackend.contract = true; }
    catch { a2aBackend.contract = false; a2aBackend.reason = 'A2A response does not satisfy the visible Agent Card list contract'; }
    delete a2aBackend.body;
    if (features.includes('a2a-write')) a2aLocalBackend = inspectNativeBackend(repo);
  }
  let driver;
  try {
    const info = driverInfo(repo, state.driverOptions);
    driver = { available: true, entry: info.entry, version: info.version, channel: info.channel };
  } catch (error) { driver = { available: false, error: error.message }; }
  const ok = Boolean(ownership && buildUnchanged && driver.available && page.available && page.body.includes('Mooc Manus'));
  return { checkedAt: new Date().toISOString(), ok, ownership, buildUnchanged,
    head: state.source.head, workingTreeDigest: state.source.digest, url: state.url,
    pageStatus: page.status, backend: api, driver, ...(llmBackend ? { llmBackend } : {}), ...(a2aBackend ? { a2aBackend } : {}), ...(a2aLocalBackend ? { a2aLocalBackend } : {}),
    eligible: { settings: Boolean(ok && api.available && api.contract), sessions: ok, files: ok,
      ...(llmBackend ? { llm: Boolean(ok && llmBackend.available && llmBackend.contract) } : {}),
      ...(a2aBackend ? { a2a: Boolean(ok && a2aBackend.available && a2aBackend.contract) } : {}) } };
}

export function viteLaunchCommand(root = repo, port = 4317) {
  const cwd = path.join(root, 'apps/tanstack-app');
  const manifestPath = path.join(cwd, 'package.json');
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  const devScript = manifest.scripts?.dev;
  const supported = typeof devScript === 'string' && devScript.trim().split(/\s+/).join(' ') === 'vite dev --port 3000';
  if (!supported) throw new Error('Unsupported tanstack-app scripts.dev; expected "vite dev --port 3000". Update the verification launcher before driving this checkout.');
  const require = createRequire(manifestPath);
  let cli;
  try { cli = path.join(path.dirname(require.resolve('vite/package.json')), 'bin/vite.js'); }
  catch { throw new Error('Local Vite package unavailable; run just install in apps first'); }
  if (!fs.existsSync(cli)) throw new Error('Local Vite JavaScript CLI unavailable; run just install in apps first');
  return { bin: process.execPath, args: [cli, 'dev', '--host', '127.0.0.1', '--port', String(port), '--strictPort'], cwd, devScript };
}

async function launch(run, port, options) {
  if (!Number.isInteger(port) || port < 1024 || port > 65535) throw new Error('Invalid port');
  const launchCommand = viteLaunchCommand(repo, port);
  if (listeners(port).length) throw new Error(`Port ${port} is already owned by another instance`);
  if (fs.existsSync(evidenceRoot)) {
    for (const directory of fs.readdirSync(evidenceRoot)) {
      const file = path.join(evidenceRoot, directory, 'instance.json');
      if (!fs.existsSync(file)) continue;
      const active = JSON.parse(fs.readFileSync(file, 'utf8'));
      if (!active.stoppedAt && processIdentity(active.pid) === active.processIdentity && active.processIdentity) {
        throw new Error(`Another verification run is active: ${active.run}. Clean it up first.`);
      }
    }
  }
  await new Promise((resolve, reject) => {
    const probe = net.createServer();
    probe.once('error', reject);
    probe.listen(port, '127.0.0.1', () => probe.close(resolve));
  });
  for (const file of ['apps/generated/pkg/package.json', 'apps/generated/types/dist/app.js']) {
    if (!fs.existsSync(path.join(repo, file))) throw new Error(`Missing ${file}; run just install in apps first`);
  }
  if (fs.existsSync(run)) throw new Error('Choose a fresh evidence directory; existing evidence is retained');
  fs.mkdirSync(run, { recursive: true });
  const source = snapshot();
  json(path.join(run, 'working-tree.json'), source);
  const log = fs.openSync(path.join(run, 'server.log'), 'a');
  // Node owns the process group directly; pnpm 12 may detach its script shells.
  const child = spawn(launchCommand.bin, launchCommand.args, { cwd: launchCommand.cwd, detached: true,
    env: { ...process.env, VITE_API_BASE_URL: '' }, stdio: ['ignore', log, log] });
  await new Promise((resolve, reject) => { child.once('spawn', resolve); child.once('error', reject); });
  fs.closeSync(log);
  child.unref();
  const state = { schema: 'verify-mooc-manus/v1', repo, run, port, url: `http://127.0.0.1:${port}`,
    pid: child.pid, processIdentity: processIdentity(child.pid), source,
    features: (options.features ?? 'settings,sessions,files').split(','),
    driverOptions: { modulePath: options['playwright-module'], channel: options.channel },
    command: [launchCommand.bin, ...launchCommand.args], cwd: launchCommand.cwd, packageDevScript: launchCommand.devScript, overrides: { VITE_API_BASE_URL: '' }, startedAt: new Date().toISOString() };
  try {
    state.members = groupMembers(state.pid);
    json(path.join(run, 'instance.json'), state);
    const deadline = Date.now() + 60000;
    while (Date.now() < deadline) {
      if (!owned(state)) throw new Error('Started process exited or ownership changed; see server.log');
      const ready = await get(state.url);
      if (ready.available && ready.body.includes('Mooc Manus')) {
        const report = await doctor(state);
        json(path.join(run, 'launch-doctor.json'), report);
        if (!report.ok) throw new Error('Instance failed doctor');
        return state;
      }
      await sleep(500);
    }
    throw new Error('Frontend readiness timeout');
  } catch (error) {
    try { json(path.join(run, 'launch-error.json'), { error: error.message }); } catch { console.error(error.message); }
    await cleanup(state);
    throw error;
  }
}

async function cleanup(state) {
  if (state.stoppedAt) return { alreadyStopped: true, evidence: state.run };
  let members = groupMembers(state.pid);
  const known = member => (state.members ?? []).some(saved => saved.pid === member.pid && saved.birth === member.birth);
  if (members.length) {
    if (owned(state)) {
      state.members = members;
      json(path.join(state.run, 'instance.json'), state);
    } else if (!members.every(known)) {
      throw new Error('Cleanup refused: group contains a process whose ownership is unproven');
    }
    process.kill(-state.pid, 'SIGTERM');
    for (let i = 0; i < 30; i++) {
      members = groupMembers(state.pid);
      if (!members.length) break;
      await sleep(100);
    }
    if (members.length) {
      if (!members.every(known)) throw new Error('Cleanup refused: remaining group ownership changed');
      process.kill(-state.pid, 'SIGKILL');
    }
  }
  for (let i = 0; i < 30 && (groupMembers(state.pid).length || listeners(state.port).length); i++) await sleep(100);
  if (groupMembers(state.pid).length || listeners(state.port).length) throw new Error('Processes or listener remain; inspect ownership and rerun cleanup');
  state.stoppedAt = new Date().toISOString();
  json(path.join(state.run, 'instance.json'), state);
  const result = { stoppedAt: state.stoppedAt, portReleased: true, remainingOwnedProcesses: [], evidencePreserved: true,
    files: fs.readdirSync(state.run).sort() };
  json(path.join(state.run, 'cleanup.json'), result);
  return result;
}

export function recordBuildConsistency(run, expectedDigest, final, report) {
  const consistent = final.digest === expectedDigest;
  json(path.join(run, 'working-tree-after.json'), { ...final, unchanged: consistent });
  const resultsFile = path.join(run, 'results.json');
  if (!fs.existsSync(resultsFile)) return report;
  const saved = report ?? JSON.parse(fs.readFileSync(resultsFile, 'utf8'));
  saved.buildUnchanged = consistent;
  if (!consistent) saved.invalidated = 'Working tree changed during drive; rerun against a stable checkout';
  for (const feature of saved.features) {
    feature.buildUnchanged = consistent;
    if (!consistent) {
      feature.invalidated = saved.invalidated;
      if (feature.status === 'passed') feature.status = 'failed';
    }
    const featureFile = path.join(run, feature.id, 'result.json');
    if (fs.existsSync(featureFile)) json(featureFile, feature);
  }
  json(resultsFile, saved);
  return saved;
}

async function drive(state, features, options) {
  const health = await doctor(state, features);
  json(path.join(state.run, 'drive-doctor.json'), health);
  if (!health.ok) throw new Error('Doctor failed; inspect drive-doctor.json before driving');
  if (fs.existsSync(path.join(state.run, 'results.json'))) throw new Error('Drive already recorded; use a new run to preserve proof');
  let result;
  try {
    result = await driveFlows({ state, health, features, modulePath: options['playwright-module'] ?? state.driverOptions.modulePath,
      channel: options.channel ?? state.driverOptions.channel, allowConfigWrite: options['allow-config-write'] === 'true' });
  } finally {
    result = recordBuildConsistency(state.run, state.source.digest, snapshot(), result);
  }
  return result;
}

async function main() {
  const args = process.argv.slice(2);
  const verb = args.shift();
  const options = {};
  while (args.length) {
    const key = args.shift();
    if (!key.startsWith('--') || !args.length) throw new Error('Options use --name value');
    options[key.slice(2)] = args.shift();
  }
  if (!['run', 'launch', 'doctor', 'drive', 'cleanup'].includes(verb)) {
    console.log('Usage: verify.mjs run|launch|doctor|drive|cleanup [--run output/playwright/verify-mooc-manus/ID] [--port 4317] [--features settings,sessions,files|settings-save|llm|llm-save|a2a|a2a-write] [--allow-config-write true] [--playwright-module /path/to/playwright] [--channel chrome]');
    process.exit(verb === '--help' || !verb ? 0 : 1);
  }
  let state;
  try {
    const features = (options.features ?? 'settings,sessions,files').split(',');
    if (!features.length || features.some(f => !['settings', 'sessions', 'files', 'settings-save', 'llm', 'llm-save', 'a2a', 'a2a-write'].includes(f))) throw new Error('Unknown feature');
    if (['launch', 'run', 'drive'].includes(verb)) requireConfigWriteAuthorization(features, options['allow-config-write'] === 'true');
    const run = resolveRunDirectory(options.run ?? path.join(evidenceRoot, new Date().toISOString().replace(/[:.]/g, '-') + '-' + randomUUID().slice(0, 8)));
    if (['doctor', 'drive', 'cleanup'].includes(verb) && !options.run) throw new Error('--run is required');
    if (verb === 'launch' || verb === 'run') state = await launch(run, Number(options.port ?? 4317), options);
    else state = readState(run);
    let result;
    if (verb === 'launch') result = { run, url: state.url, pid: state.pid };
    if (verb === 'doctor') { result = await doctor(state, options.features ? features : undefined); if (!result.ok) process.exitCode = 1; }
    if (verb === 'cleanup') result = await cleanup(state);
    if (verb === 'drive' || verb === 'run') {
      try {
        result = await drive(state, features, options);
        if (result.features.some(f => f.status === 'failed')) process.exitCode = 1;
        else if (result.features.some(f => f.status === 'blocked')) process.exitCode = 2;
      } finally { if (verb === 'run') await cleanup(state); }
    }
    console.log(JSON.stringify({ run: state.run, ...result }, null, 2));
  } catch (error) {
    if (state && !state.stoppedAt && verb !== 'doctor') {
      try { await cleanup(state); } catch (cleanupError) { console.error(`Cleanup needs attention: ${cleanupError.message}`); }
    }
    console.error(error.stack);
    process.exitCode = 1;
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
