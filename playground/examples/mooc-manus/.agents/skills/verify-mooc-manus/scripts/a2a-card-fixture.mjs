#!/usr/bin/env node
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { validFixtureIdentity } from './a2a-policy.mjs';

export function nativeBackendVerdict({ pids, executable, cwd, repo, processIdentity }) {
  const ok = pids.length === 1 && Number.isSafeInteger(pids[0]) && pids[0] > 1
    && path.isAbsolute(executable ?? '') && path.basename(executable ?? '') === 'server-cli'
    && cwd === path.join(repo, 'server') && Boolean(processIdentity);
  return { ok, ...(ok ? { pid: pids[0], executable, cwd, processIdentity } : {}),
    reason: ok ? 'Native local server-cli in this checkout; loopback fixture is reachable in the same host namespace'
      : 'A2A write requires one native local server-cli listener in this checkout server directory; container or unknown ownership is blocked' };
}

export function inspectNativeBackend(repo, port = 5150) {
  const run = (bin, args) => {
    const result = spawnSync(bin, args, { encoding: 'utf8', timeout: 4000 });
    if (result.error || result.status !== 0) throw new Error('Native backend process inspection unavailable');
    return result.stdout.trim();
  };
  try {
    const pids = [...new Set(run('lsof', ['-nP', '-t', `-iTCP:${port}`, '-sTCP:LISTEN']).split(/\s+/).map(Number))];
    if (pids.length !== 1) return nativeBackendVerdict({ pids, repo });
    const pid = String(pids[0]);
    const executable = run('ps', ['-p', pid, '-o', 'comm=']);
    const processIdentity = run('ps', ['-p', pid, '-o', 'lstart=']);
    const cwd = run('lsof', ['-a', '-p', pid, '-d', 'cwd', '-Fn']).split('\n').find(line => line.startsWith('n'))?.slice(1);
    return nativeBackendVerdict({ pids, executable, processIdentity, cwd, repo });
  } catch { return { ok: false, reason: 'Native backend process inspection unavailable; A2A write blocked' }; }
}

const cardFor = nonce => ({ name: `Verification Agent ${nonce}`, description: `Local controlled Agent Card ${nonce}`,
  capabilities: { streaming: false, push_notifications: false }, defaultInputModes: ['text'], defaultOutputModes: ['text'] });

export async function startCardFixture(directory, saved) {
  fs.mkdirSync(directory, { recursive: true });
  const nonce = saved?.nonce ?? randomUUID();
  let descriptor;
  if (saved && (!validFixtureIdentity(saved) || saved.schema !== 'a2a-card-fixture/v1'
    || JSON.stringify(saved.card) !== JSON.stringify(cardFor(nonce)))) throw new Error('Invalid fixture recovery descriptor');
  const observations = [];
  const server = http.createServer((request, response) => {
    const requestUrl = new URL(request.url, 'http://127.0.0.1');
    const allowed = request.method === 'GET' && request.url === `/${nonce}/.well-known/agent-card.json`;
    const observation = { at: new Date().toISOString(), method: request.method, path: requestUrl.pathname, servedCard: allowed };
    observations.push(observation);
    fs.appendFileSync(path.join(directory, 'fixture-requests.jsonl'), JSON.stringify(observation) + '\n');
    response.writeHead(allowed ? 200 : request.method === 'GET' ? 404 : 405, { 'Content-Type': 'application/json' });
    response.end(allowed ? JSON.stringify(descriptor.card) : JSON.stringify({ error: 'Fixture only serves its Agent Card; invocation is disabled' }));
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(saved ? Number(new URL(saved.baseUrl).port) : 0, '127.0.0.1', resolve);
  });
  const baseUrl = `http://127.0.0.1:${server.address().port}/${nonce}`;
  const name = `Verification Agent ${nonce}`, description = `Local controlled Agent Card ${nonce}`;
  descriptor = { schema: 'a2a-card-fixture/v1', nonce, baseUrl, name, description,
    card: cardFor(nonce) };
  try {
    fs.writeFileSync(path.join(directory, 'fixture.json'), JSON.stringify(descriptor, null, 2) + '\n');
    fs.appendFileSync(path.join(directory, 'fixture-lifecycle.jsonl'), JSON.stringify({ startedAt: new Date().toISOString(), pid: process.pid, baseUrl, restarted: Boolean(saved) }) + '\n');
  } catch (error) {
    const stopped = new Promise(resolve => server.close(resolve));
    server.closeAllConnections();
    await stopped;
    throw error;
  }
  const close = async () => {
    if (!server.listening) return;
    const finished = new Promise(resolve => server.close(resolve));
    server.closeAllConnections();
    await finished;
    fs.appendFileSync(path.join(directory, 'fixture-lifecycle.jsonl'), JSON.stringify({ stoppedAt: new Date().toISOString(), pid: process.pid }) + '\n');
  };
  return { descriptor, observations, close };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv[2] !== 'serve' || process.argv[3] !== '--fixture' || !process.argv[4]) throw new Error('Usage: a2a-card-fixture.mjs serve --fixture /absolute/run/a2a-write/fixture.json');
  const file = path.resolve(process.argv[4]);
  const fixture = await startCardFixture(path.dirname(file), JSON.parse(fs.readFileSync(file, 'utf8')));
  console.log(JSON.stringify({ pid: process.pid, baseUrl: fixture.descriptor.baseUrl, scope: 'Agent Card only; no invocation; stop with Ctrl+C' }));
  for (const signal of ['SIGINT', 'SIGTERM']) process.once(signal, () => { void fixture.close(); });
}
