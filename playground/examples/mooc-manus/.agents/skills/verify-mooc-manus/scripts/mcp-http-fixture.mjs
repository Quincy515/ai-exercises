#!/usr/bin/env node
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { validMcpFixture } from './mcp-policy.mjs';

export async function startMcpFixture(directory, saved) {
  if (saved && !validMcpFixture(saved)) throw new Error('Invalid MCP fixture recovery descriptor');
  fs.mkdirSync(directory, { recursive: true });
  const nonce = saved?.nonce ?? randomUUID();
  let descriptor;
  const observations = [];
  const pendingHandlers = new Set();
  let handlerFailed = false, closePromise;
  const observe = entry => {
    const safe = { at: new Date().toISOString(), ...entry };
    observations.push(safe);
    fs.appendFileSync(path.join(directory, 'fixture-requests.jsonl'), JSON.stringify(safe) + '\n');
  };
  const handle = async (request, response) => {
    const reply = (status, data) => {
      if (response.destroyed || response.writableEnded) return;
      response.writeHead(status, { 'Content-Type': 'application/json' });
      response.end(data === undefined ? '' : JSON.stringify(data));
    };
    if (request.url !== `/${nonce}/mcp` || request.method !== 'POST') {
      observe({ httpMethod: request.method, operation: 'unsupported-http', accepted: false });
      reply(405, { error: 'Only controlled MCP protocol POSTs are supported' }); return;
    }
    let value;
    try {
      let size = 0; const chunks = [];
      for await (const chunk of request) { size += chunk.length; if (size > 64 * 1024) throw new Error('too large'); chunks.push(chunk); }
      value = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    } catch {
      observe({ httpMethod: 'POST', operation: 'invalid-json', accepted: false }); reply(400, { error: 'Invalid protocol request' }); return;
    }
    const method = value?.method;
    const allowed = value?.jsonrpc === '2.0' && ['initialize', 'notifications/initialized', 'tools/list'].includes(method);
    // A method itself is untrusted input, so never write arbitrary strings/params.
    observe({ httpMethod: 'POST', operation: allowed ? method : method === 'tools/call' ? 'tools/call' : 'unsupported-rpc', accepted: allowed });
    if (!allowed) { reply(403, { jsonrpc: '2.0', id: value?.id ?? null, error: { code: -32601, message: 'Tool invocation is disabled' } }); return; }
    if (method === 'notifications/initialized') { reply(202); return; }
    if (!['string', 'number'].includes(typeof value.id)) { reply(400, { error: 'Request ID required' }); return; }
    const result = method === 'initialize'
      ? { protocolVersion: ['2024-11-05', '2025-03-26', '2025-06-18'].includes(value.params?.protocolVersion) ? value.params.protocolVersion : '2025-03-26',
        capabilities: { tools: {} }, serverInfo: { name: 'Controlled MCP verification', version: '1.0.0' } }
      : { tools: [{ name: descriptor.toolName, description: 'Verification metadata only; invocation disabled',
        inputSchema: { type: 'object', properties: {}, additionalProperties: false } }] };
    // Stateless JSON responses avoid a background SSE and session cleanup.
    reply(200, { jsonrpc: '2.0', id: value.id, result });
  };
  const server = http.createServer((request, response) => {
    response.on('error', () => { handlerFailed = true; });
    // Node does not await an async request listener. Own its lifetime explicitly
    // so close waits for aborted bodies and their final evidence writes.
    const task = handle(request, response).catch(() => {
      handlerFailed = true;
      response.destroy();
    });
    pendingHandlers.add(task);
    task.then(() => pendingHandlers.delete(task), () => { handlerFailed = true; pendingHandlers.delete(task); });
  });
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(saved?.port ?? 0, '127.0.0.1', resolve); });
  descriptor = { schema: 'mcp-http-fixture/v1', nonce, port: server.address().port,
    serverName: `verification-mcp-${nonce}`, toolName: `verify_${nonce.replaceAll('-', '')}` };
  const close = () => {
    closePromise ??= (async () => {
      const stopped = new Promise((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
      server.closeAllConnections();
      let closeFailed = false;
      try { await stopped; } catch { closeFailed = true; }
      while (pendingHandlers.size) await Promise.allSettled([...pendingHandlers]);
      fs.appendFileSync(path.join(directory, 'fixture-lifecycle.jsonl'), JSON.stringify({ stoppedAt: new Date().toISOString(), pid: process.pid, handlerFailed }) + '\n');
      if (closeFailed || handlerFailed) throw new Error('MCP fixture shutdown failed; raw diagnostics excluded');
    })();
    return closePromise;
  };
  try {
    fs.writeFileSync(path.join(directory, 'fixture.json'), JSON.stringify(descriptor, null, 2) + '\n');
    fs.appendFileSync(path.join(directory, 'fixture-lifecycle.jsonl'), JSON.stringify({ startedAt: new Date().toISOString(), pid: process.pid, port: descriptor.port, restarted: Boolean(saved) }) + '\n');
  } catch (error) { await close(); throw error; }
  return { descriptor, observations, close };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv[2] !== 'serve' || process.argv[3] !== '--fixture' || !process.argv[4]) throw new Error('Usage: mcp-http-fixture.mjs serve --fixture /absolute/run/mcp-write/fixture.json');
  const file = path.resolve(process.argv[4]);
  const fixture = await startMcpFixture(path.dirname(file), JSON.parse(fs.readFileSync(file, 'utf8')));
  console.log(JSON.stringify({ pid: process.pid, serverName: fixture.descriptor.serverName, port: fixture.descriptor.port, scope: 'Initialization and tools/list only; invocation disabled' }));
  for (const signal of ['SIGINT', 'SIGTERM']) process.once(signal, () => {
    void fixture.close().catch(() => { console.error('MCP fixture shutdown failed; raw diagnostics excluded'); process.exitCode = 1; });
  });
}
