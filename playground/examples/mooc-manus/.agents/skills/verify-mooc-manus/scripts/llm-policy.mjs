#!/usr/bin/env node
import assert from 'node:assert/strict';

export const llmPath = '/api/app_configs/llm';
export const llmFields = ['base_url', 'model_name', 'temperature', 'max_tokens'];
const nullable = (value, check) => value === null || check(value);

export function validLlmPayload(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    && Object.keys(value).length === llmFields.length && llmFields.every(key => Object.hasOwn(value, key))
    && nullable(value.base_url, value => typeof value === 'string')
    && nullable(value.model_name, value => typeof value === 'string')
    && nullable(value.temperature, value => Number.isFinite(value) && value >= -2 && value <= 2)
    && nullable(value.max_tokens, value => Number.isSafeInteger(value) && value >= 0);
}

export function llmPayload(config) {
  const values = Object.fromEntries(llmFields.map(key => [key, config?.[key]]));
  assert.ok(validLlmPayload(values), 'LLM public fields do not satisfy the safe configuration contract');
  return values;
}

// Every persisted response is built from this whitelist. Raw response text and
// unknown properties (including legacy api_key) never enter proof artifacts.
export function readLlmConfig(text) {
  let raw;
  try { raw = JSON.parse(text); } catch { throw new Error('LLM configuration response is not valid JSON'); }
  if (raw?.max_tokens !== null && typeof raw?.max_tokens === 'number' && !Number.isSafeInteger(raw.max_tokens)) {
    const error = new Error('LLM max_tokens exceeds safe JavaScript integer handling; verification blocked');
    error.code = 'LLM_UNSAFE_MAX_TOKENS';
    throw error;
  }
  const values = llmPayload(raw);
  assert.ok(typeof raw?.api_key_configured === 'boolean', 'LLM key configuration status must be boolean');
  return { ...values, api_key_configured: raw.api_key_configured };
}

export function sameLlmConfig(left, right) {
  if (!left || !right) return false;
  const leftPayload = Object.fromEntries(llmFields.map(key => [key, left[key]]));
  const rightPayload = Object.fromEntries(llmFields.map(key => [key, right[key]]));
  return validLlmPayload(leftPayload) && validLlmPayload(rightPayload)
    && typeof left.api_key_configured === 'boolean' && left.api_key_configured === right.api_key_configured
    && llmFields.every(key => left[key] === right[key]);
}

export function allowedLlmWrite({ feature, allowConfigWrite, baseUrl, url, method, payload, expected }) {
  return feature === 'llm-save' && allowConfigWrite === true && method === 'POST'
    && url === new URL(llmPath, baseUrl).href && validLlmPayload(payload) && validLlmPayload(expected)
    && llmFields.every(key => payload[key] === expected[key]);
}

export function llmRestoreDecision(current, original, target) {
  assert.ok(sameLlmConfig(original, original) && sameLlmConfig(target, target), 'LLM restore requires validated public snapshots');
  if (sameLlmConfig(current, original)) return 'already-restored';
  if (sameLlmConfig(current, target)) return 'restore';
  throw new Error('LLM cleanup conflict: public configuration or key status changed; preserve original-config.json for manual reconciliation');
}

export function llmTarget(original) {
  return { ...original, max_tokens: original.max_tokens === null ? 1
    : original.max_tokens === Number.MAX_SAFE_INTEGER ? original.max_tokens - 1 : original.max_tokens + 1 };
}

export function llmInputMatches(key, actual, expected) {
  if (!llmFields.includes(key) || typeof actual !== 'string') return false;
  if (expected === null) return actual === '';
  if (key !== 'temperature') return actual === String(expected);
  const numeric = Number(actual);
  return actual.trim() !== '' && Number.isFinite(numeric) && Number.isFinite(expected)
    && Number.isFinite(Math.fround(numeric)) && Number.isFinite(Math.fround(expected))
    && Math.fround(numeric) === Math.fround(expected);
}
