#!/usr/bin/env node
import assert from 'node:assert/strict';

export const a2aPath = '/api/app_configs/a2a-servers';
const rowKeys = ['id', 'name', 'description', 'input_modes', 'output_modes', 'streaming', 'push_notifications', 'enabled'];
const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const exactKeys = (value, keys) => value && typeof value === 'object' && !Array.isArray(value)
  && Object.keys(value).length === keys.length && keys.every(key => Object.hasOwn(value, key));

export function readA2aList(text) {
  let raw;
  try { raw = JSON.parse(text); } catch { throw new Error('A2A list response is not valid JSON'); }
  assert.ok(Array.isArray(raw?.a2a_servers), 'A2A response must contain an a2a_servers array');
  const result = raw.a2a_servers.map(row => {
    assert.ok(row && ['id', 'name', 'description'].every(key => typeof row[key] === 'string')
      && row.id.length > 0 && ['streaming', 'push_notifications', 'enabled'].every(key => typeof row[key] === 'boolean')
      && ['input_modes', 'output_modes'].every(key => Array.isArray(row[key]) && row[key].every(value => typeof value === 'string')),
    'A2A list row does not satisfy the public contract');
    return Object.fromEntries(rowKeys.map(key => [key, row[key]]));
  });
  assert.equal(new Set(result.map(row => row.id)).size, result.length, 'A2A list contains duplicate IDs');
  return result;
}

export function validFixtureIdentity(fixture) {
  if (!fixture || !uuid.test(fixture.nonce) || fixture.name !== `Verification Agent ${fixture.nonce}`
    || fixture.description !== `Local controlled Agent Card ${fixture.nonce}`) return false;
  try {
    const url = new URL(fixture.baseUrl);
    return url.protocol === 'http:' && url.hostname === '127.0.0.1' && Number(url.port) >= 1024
      && Number(url.port) <= 65535 && url.pathname === '/' + fixture.nonce && !url.search && !url.hash
      && !url.username && !url.password;
  } catch { return false; }
}

export function findOwnedA2a(list, baseline, fixture) {
  assert.ok(validFixtureIdentity(fixture), 'A2A ownership requires the recorded local fixture identity');
  const baselineIds = new Set(baseline.map(row => row.id));
  const matches = list.filter(row => row.name === fixture.name && row.description === fixture.description && !baselineIds.has(row.id));
  assert.ok(matches.length <= 1, 'Multiple A2A records match this run; ownership is ambiguous');
  if (!matches.length) return null;
  assert.ok(uuid.test(matches[0].id), 'Owned A2A record ID must be a server-issued UUID');
  return matches[0];
}

// Pin the first server-issued ID for the entire run, including recovery reads.
export function createA2aOwnership(baseline, fixture, onConfirm = () => {}) {
  let knownId = null;
  return {
    get id() { return knownId; },
    find(list) {
      const candidate = findOwnedA2a(list, baseline, fixture);
      if (knownId && candidate && candidate.id !== knownId) {
        const error = new Error('A2A owned ID changed; preserve the first record and reconcile manually');
        error.code = 'A2A_OWNERSHIP_CONFLICT';
        error.candidate = candidate;
        throw error;
      }
      if (candidate && knownId === null) {
        knownId = candidate.id;
        onConfirm(candidate);
      }
      return candidate;
    },
  };
}

export function sameVisibleA2a(left, right) {
  const canonical = list => [...list].sort((a, b) => a.id.localeCompare(b.id));
  return JSON.stringify(canonical(left)) === JSON.stringify(canonical(right));
}

export function allowedA2aWrite({ feature, allowConfigWrite, baseUrl, url, method, payload, body, expected }) {
  if (feature !== 'a2a-write' || allowConfigWrite !== true || method !== 'POST'
    || !expected || !validFixtureIdentity(expected.fixture) || !Array.isArray(expected.baseline)) return false;
  const endpoint = new URL(a2aPath, baseUrl).href;
  if (expected.action === 'create') return url === endpoint && expected.createAttempted === false
    && exactKeys(payload, ['base_url']) && payload.base_url === expected.fixture.baseUrl;
  const owned = expected.owned;
  if (!owned || expected.knownId !== owned.id || !uuid.test(owned.id) || owned.name !== expected.fixture.name || owned.description !== expected.fixture.description
    || expected.baseline.some(row => row.id === owned.id)) return false;
  if (expected.action === 'enabled') return url === `${endpoint}/${owned.id}/enabled`
    && exactKeys(payload, ['enabled']) && typeof expected.enabled === 'boolean' && payload.enabled === expected.enabled;
  if (expected.action === 'delete') return url === `${endpoint}/${owned.id}/delete` && (body === null || body === '');
  return false;
}

export function a2aRecoveryStatus({ journal, baseline, current, fixture, knownId, createAttempted, deleteConfirmed }) {
  const unknown = journal.entries.some(entry => entry.outcome !== 'response-received');
  const owned = current ? findOwnedA2a(current, baseline, fixture) : null;
  if (knownId && owned && owned.id !== knownId) throw new Error('A2A owned ID changed; manual reconciliation required');
  return { decision: unknown ? 'outcome-unknown' : deleteConfirmed ? 'delete-confirmed' : createAttempted ? 'manual-cleanup-required' : 'no-write',
    ownedId: owned?.id ?? knownId ?? null, visibleOwned: Boolean(owned),
    manualCleanupRequired: createAttempted && (!deleteConfirmed || unknown),
    ...(deleteConfirmed && !unknown && current ? { visibleListRestored: !owned && sameVisibleA2a(current, baseline) } : {}) };
}
