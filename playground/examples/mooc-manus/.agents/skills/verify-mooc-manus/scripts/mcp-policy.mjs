#!/usr/bin/env node
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';

export const mcpPath = '/api/app_configs/mcp-servers';
const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const exact = (value, keys) => object(value) && Object.keys(value).length === keys.length && keys.every(key => Object.hasOwn(value, key));
const validTransport = value => ['stdio', 'streamable_http'].includes(value);
const decode = text => { try { return JSON.parse(text); } catch { throw new Error('MCP response is not valid JSON; raw body excluded'); } };

export function readMcpList(text) {
  const raw = decode(text);
  assert.ok(Array.isArray(raw?.mcp_servers), 'MCP response must contain the public server list');
  const rows = raw.mcp_servers.map(row => {
    assert.ok(object(row) && typeof row.server_name === 'string' && row.server_name.length > 0
      && typeof row.enabled === 'boolean' && validTransport(row.transport)
      && Array.isArray(row.tools) && row.tools.every(tool => typeof tool === 'string'), 'Invalid public MCP server metadata');
    return { server_name: row.server_name, transport: row.transport, enabled: row.enabled, tools: row.tools };
  });
  assert.equal(new Set(rows.map(row => row.server_name)).size, rows.length, 'MCP list has duplicate server names');
  return rows;
}

export function validMcpFixture(fixture) {
  return fixture?.schema === 'mcp-http-fixture/v1' && uuid.test(fixture.nonce)
    && fixture.serverName === `verification-mcp-${fixture.nonce}`
    && fixture.toolName === `verify_${fixture.nonce.replaceAll('-', '')}`
    && Number.isInteger(fixture.port) && fixture.port >= 1024 && fixture.port <= 65535;
}

export function mcpFixtureUrl(fixture) {
  assert.ok(validMcpFixture(fixture), 'Invalid local MCP fixture descriptor');
  return `http://127.0.0.1:${fixture.port}/${fixture.nonce}/mcp`;
}

export const mcpFixtureFingerprint = fixture => createHash('sha256').update(mcpFixtureUrl(fixture)).digest('hex');
const description = (fixture, revision) => `Controlled MCP verification ${fixture.nonce} ${revision}`;

export function mcpFixturePayload(fixture, revision = 'initial') {
  assert.ok(['initial', 'updated'].includes(revision), 'Unsupported fixture revision');
  return { mcpServers: { [fixture.serverName]: { transport: 'streamable_http', enabled: true,
    description: description(fixture, revision), url: mcpFixtureUrl(fixture) } } };
}

// All other configured servers may contain secrets. Inspect only metadata and
// the owned fixture in memory; never return their full objects or the raw text.
export function readMcpWriteResponse(text, { fixture, action, enabled = true, revision = 'initial' }) {
  const raw = decode(text);
  assert.ok(object(raw?.mcpServers), 'MCP write response must contain the configuration map');
  const metadata = Object.entries(raw.mcpServers).map(([server_name, config]) => {
    assert.ok(object(config) && validTransport(config.transport) && typeof config.enabled === 'boolean', 'Invalid MCP write metadata');
    return { server_name, transport: config.transport, enabled: config.enabled };
  });
  assert.ok(validMcpFixture(fixture), 'MCP confirmation requires a controlled fixture');
  if (action === 'delete') {
    assert.ok(!Object.hasOwn(raw.mcpServers, fixture.serverName), 'MCP deletion response still contains the owned name');
  } else {
    const own = raw.mcpServers[fixture.serverName];
    assert.ok(object(own) && own.transport === 'streamable_http' && own.enabled === enabled
      && own.url === mcpFixtureUrl(fixture) && own.description === description(fixture, revision)
      && ['env', 'headers', 'args', 'command'].every(key => own[key] == null), 'Owned MCP configuration changed; raw fields excluded');
  }
  return { mcp_servers: metadata, ...(action === 'delete' ? {} : { ownedFingerprint: mcpFixtureFingerprint(fixture) }) };
}

export function sameMcpList(left, right) {
  const normalize = rows => [...rows].map(row => ({ server_name: row.server_name, transport: row.transport,
    enabled: row.enabled, tools: [...row.tools].sort() })).sort((a, b) => a.server_name.localeCompare(b.server_name));
  return JSON.stringify(normalize(left)) === JSON.stringify(normalize(right));
}

export function allowedMcpWrite({ feature, allowConfigWrite, baseUrl, url, method, payload, body, expected }) {
  if (feature !== 'mcp-write' || allowConfigWrite !== true || method !== 'POST' || !expected
    || !validMcpFixture(expected.fixture) || !Array.isArray(expected.baseline)
    || expected.baseline.some(row => row.server_name === expected.fixture.serverName)) return false;
  const { fixture, action } = expected;
  const endpoint = new URL(mcpPath, baseUrl).href;
  if (action !== 'create' && expected.ownedFingerprint !== mcpFixtureFingerprint(fixture)) return false;
  if (action === 'create' || action === 'update') {
    if (action === 'create' && expected.createAttempted !== false) return false;
    if (!['initial', 'updated'].includes(expected.revision) || url !== endpoint || !exact(payload, ['mcpServers'])
      || !exact(payload.mcpServers, [fixture.serverName])) return false;
    const config = payload.mcpServers[fixture.serverName];
    return object(config) && Object.keys(config).every(key => ['transport', 'enabled', 'description', 'url', 'env', 'headers', 'args', 'command'].includes(key))
      && config.transport === 'streamable_http' && config.enabled === true && config.url === mcpFixtureUrl(fixture)
      && config.description === description(fixture, expected.revision)
      && ['env', 'headers', 'args', 'command'].every(key => config[key] == null);
  }
  if (action === 'enabled') return url === `${endpoint}/${encodeURIComponent(fixture.serverName)}/enabled`
    && exact(payload, ['enabled']) && typeof expected.enabled === 'boolean' && payload.enabled === expected.enabled;
  if (action === 'delete') return url === `${endpoint}/${encodeURIComponent(fixture.serverName)}/delete` && (body === null || body === '');
  return false;
}

// Authorization has already validated the raw request. Persist public metadata
// only, including for our own controlled URL, to keep one evidence rule.
export function mcpWriteEvidence(expected) {
  return { action: expected.action, server_name: expected.fixture.serverName, transport: 'streamable_http',
    ...(expected.action === 'delete' ? {} : { enabled: expected.action === 'enabled' ? expected.enabled : true }) };
}

export function mcpRecoveryStatus({ journal, baseline, current, fixture, ownedFingerprint, createAttempted, deleteConfirmed }) {
  const unknown = journal.entries.some(entry => entry.outcome !== 'response-received');
  return { decision: unknown ? 'outcome-unknown' : deleteConfirmed ? 'delete-confirmed' : createAttempted ? 'manual-cleanup-required' : 'no-write',
    plannedName: fixture.serverName, ownedName: ownedFingerprint ? fixture.serverName : null,
    ownershipConfirmed: ownedFingerprint === mcpFixtureFingerprint(fixture),
    manualCleanupRequired: createAttempted && (!deleteConfirmed || unknown),
    ...(deleteConfirmed && !unknown && current ? { visibleListRestored: sameMcpList(current, baseline) } : {}) };
}
