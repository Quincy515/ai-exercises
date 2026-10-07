#!/usr/bin/env node
import assert from 'node:assert/strict';

export const configKeys = ['max_iterations', 'max_retries', 'max_search_results'];
export const configPath = '/api/app_configs/agent';

export function requireConfigWriteAuthorization(features, permission) {
  if (features.includes('settings-save') && permission !== true) {
    throw new Error('settings-save requires explicit --allow-config-write true for this launch/run/drive');
  }
}

export function validConfig(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    && Object.keys(value).length === configKeys.length
    && configKeys.every(key => Object.hasOwn(value, key) && Number.isSafeInteger(value[key]))
    && value.max_iterations >= 1 && value.max_iterations <= 999
    && value.max_retries >= 2 && value.max_retries <= 9
    && value.max_search_results >= 2 && value.max_search_results <= 29;
}

export function sameConfig(left, right) {
  return validConfig(left) && validConfig(right) && configKeys.every(key => left[key] === right[key]);
}

// The browser guard and exceptional cleanup share the exact endpoint/payload policy.
export function allowedConfigWrite({ feature, allowConfigWrite, baseUrl, url, method, payload, expected }) {
  return feature === 'settings-save' && allowConfigWrite === true && method === 'POST'
    && url === new URL(configPath, baseUrl).href && sameConfig(payload, expected);
}

export function restoreDecision(current, original, target) {
  assert.ok(validConfig(original) && validConfig(target), 'Original and target must satisfy the live write contract');
  if (sameConfig(current, original)) return 'already-restored';
  if (sameConfig(current, target)) return 'restore';
  throw new Error('Config cleanup conflict: server has a third-party value; preserve original-config.json and resolve manually');
}

// A transport failure can happen after the server accepted the POST. Unknown is
// sticky: a late response is retained as evidence and never upgrades this run.
export class ConfigWriteJournal {
  entries = [];
  constructor(persist = () => {}) { this.persist = persist; }
  begin({ url, payload, source = 'ui' }) {
    const entry = { id: this.entries.length + 1, method: 'POST', url, payload, source,
      outcome: 'pending', authorizedAt: new Date().toISOString() };
    this.entries.push(entry);
    this.persist(this.entries);
    return entry;
  }
  headers(entry, status) {
    entry.status = status;
    this.persist(this.entries);
  }
  complete(entry, status = entry.status) {
    entry.status = status;
    entry.responseFinishedAt = new Date().toISOString();
    if (entry.outcome === 'pending') entry.outcome = 'response-received';
    else if (entry.outcome === 'outcome-unknown') entry.lateResponseReceived = true;
    this.persist(this.entries);
  }
  unknown(entry, reason) {
    if (entry.outcome !== 'pending') return;
    entry.outcome = 'outcome-unknown';
    entry.reason = reason;
    entry.unknownAt = new Date().toISOString();
    this.persist(this.entries);
  }
  finalize(reason = 'Verification ended before a complete POST response was observed') {
    for (const entry of this.entries) this.unknown(entry, reason);
  }
}

export async function withinPostDeadline(operation, journal, timeoutMs = 15000) {
  let timer;
  try {
    return await Promise.race([operation(), new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error('POST response deadline exceeded; server outcome is unknown')), timeoutMs);
    })]);
  } catch (error) {
    journal.finalize(error.message);
    throw error;
  } finally { clearTimeout(timer); }
}

export async function restoreConfigAfterWrites({ journal, original, target, read, write, progress = {} }) {
  const unsettled = journal.entries.filter(entry => entry.outcome !== 'response-received');
  if (unsettled.length) {
    progress.decision = 'outcome-unknown';
    progress.unresolvedWriteIds = unsettled.map(entry => entry.id);
    throw new Error('Config cleanup outcome-unknown: POST may still commit; automatic restore and restored=true are withheld. Preserve original-config.json and post-outcomes.json for reconciliation');
  }
  const current = await read('最终恢复检查');
  progress.before = current;
  progress.decision = restoreDecision(current, original, target);
  if (progress.decision === 'restore') {
    await write(original);
    progress.compensatingWrite = true;
  }
  const after = await read('恢复后GET确认');
  assert.ok(sameConfig(after, original), 'Final GET differs from original values');
  progress.after = after;
  progress.restored = true;
  return progress;
}

// Fetch the real backend response once, with redirects disabled. Continuing the
// browser request would let 307/308 forward its POST beyond the approved URL.
export async function forwardAuthorizedConfigWrite(route, journal, entry) {
  try {
    const response = await route.fetch({ maxRedirects: 0, maxRetries: 0, timeout: 10000 });
    journal.headers(entry, response.status());
    if (response.status() >= 300 && response.status() < 400) {
      journal.unknown(entry, `POST redirect ${response.status()} blocked before following its Location`);
      await route.abort('blockedbyclient');
      return;
    }
    await route.fulfill({ response });
    journal.complete(entry, response.status());
  } catch (error) {
    journal.unknown(entry, error.message);
    try { await route.abort('blockedbyclient'); } catch { /* The browser may already have closed the request. */ }
  }
}
