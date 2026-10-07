#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { allowedConfigWrite, requireConfigWriteAuthorization, restoreConfigAfterWrites, withinPostDeadline } from './config-write.mjs';
import { llmFields, llmInputMatches, llmPath, llmPayload, llmRestoreDecision, llmTarget, readLlmConfig, sameLlmConfig } from './llm-policy.mjs';

export async function driveLlm({ feature, page, state, health, result, step, capture, record, directory, writeGuard, allowConfigWrite, goto }) {
  const writing = feature === 'llm-save';
  if (writing) requireConfigWriteAuthorization(['llm-save'], allowConfigWrite);
  const endpoint = state.url + llmPath;
  const saveJson = (name, value) => fs.writeFileSync(path.join(directory, name), JSON.stringify(value, null, 2) + '\n');
  const dialog = page.getByRole('dialog', { name: 'MoocManus 设置' });
  const form = dialog.locator('#llm-config-form');
  const routes = ['/', '/sessions/1', '/schedules', '/library'];
  const responses = [];
  let original, target, writeAttempted = false;
  result.unverified = ['密钥替换与真实模型调用', 'Electron', '移动端', '并发第三方写入的原子冲突检测'];
  if (writing) result.unverifiedEntryPoints = routes.slice(1).map(route => route + ' → 模型提供商 → 保存');
  const watch = method => {
    const pending = page.waitForResponse(response => response.url() === endpoint && response.request().method() === method);
    pending.catch(() => {});
    return pending;
  };
  const readResponse = async (response, action, method = 'GET') => {
    assert.equal(response.status(), 200, `${action} must return HTTP 200`);
    const values = readLlmConfig(await response.text());
    responses.push({ action, method, status: response.status(), values });
    saveJson('api-responses.json', responses);
    return values;
  };
  const checkInputs = async expected => {
    for (const key of llmFields) {
      await page.waitForFunction(({ key, expected }) => {
        const actual = document.querySelector('#llm-config-form #' + key)?.value;
        if (typeof actual !== 'string') return false;
        if (expected === null) return actual === '';
        if (key !== 'temperature') return actual === String(expected);
        const numeric = Number(actual);
        return actual.trim() !== '' && Number.isFinite(numeric) && Number.isFinite(expected)
          && Number.isFinite(Math.fround(numeric)) && Number.isFinite(Math.fround(expected))
          && Math.fround(numeric) === Math.fround(expected);
      }, { key, expected: expected[key] });
      assert.ok(llmInputMatches(key, await form.locator('#' + key).inputValue(), expected[key]), `LLM ${key} input differs from the server value`);
      assert.equal(await form.locator('#' + key).isEditable(), true);
    }
    const key = form.locator('#api_key');
    assert.equal(await key.getAttribute('type'), 'password');
    assert.ok(await key.inputValue() === '', 'API key input must remain empty; value omitted from evidence');
    assert.ok(await key.getAttribute('placeholder') === (expected.api_key_configured
      ? '已配置，留空保留原密钥' : '填写新的 API 密钥'), 'LLM key placeholder must reflect configuration status');
  };
  const open = async () => {
    const pending = watch('GET');
    await page.getByRole('button', { name: '打开设置', exact: true }).click();
    await dialog.getByRole('button', { name: '模型提供商', exact: true }).click();
    const values = await readResponse(await pending, '打开模型配置');
    await checkInputs(values);
    await form.getByRole('status').filter({ hasText: '已读取服务器模型配置' }).waitFor();
    assert.equal(await dialog.getByRole('button', { name: '保存', exact: true }).isDisabled(), true);
    return values;
  };
  const close = async () => {
    await dialog.getByRole('button', { name: '取消', exact: true }).click();
    await dialog.waitFor({ state: 'hidden' });
  };
  const readCurrent = async action => {
    const response = await page.request.get(endpoint, { timeout: 10000, maxRedirects: 0 });
    const values = await readResponse(response, action);
    record({ action, method: 'GET', values });
    return values;
  };
  const saveThroughUi = async (expected, previous, action) => {
    assert.ok(sameLlmConfig(await readCurrent(action + '前检查'), previous), 'LLM configuration changed before save; refusing to overwrite');
    await form.locator('#temperature').fill(expected.temperature === null ? '' : String(expected.temperature));
    await form.locator('#max_tokens').fill(expected.max_tokens === null ? '' : String(expected.max_tokens));
    assert.ok(await form.locator('#api_key').inputValue() === '', 'LLM verification must preserve an empty key input');
    writeGuard.expected = llmPayload(expected);
    const pending = watch('POST');
    writeAttempted = true;
    try {
      await dialog.getByRole('button', { name: '保存', exact: true }).click();
      await withinPostDeadline(async () => {
        const response = await pending;
        const failure = await response.finished();
        if (failure) throw new Error('LLM POST transport did not finish');
        const entry = writeGuard.requests.get(response.request());
        assert.ok(entry, 'LLM response must match an authorized request');
        writeGuard.journal.complete(entry, response.status());
        assert.ok(sameLlmConfig(await readResponse(response, action, 'POST'), expected), 'LLM saved fields or key status differ from the expected values');
      }, writeGuard.journal);
      await form.getByRole('status').filter({ hasText: '模型配置保存成功' }).waitFor();
      await checkInputs(expected);
      assert.equal(await dialog.getByRole('button', { name: '保存', exact: true }).isDisabled(), true);
    } finally { writeGuard.expected = null; }
  };
  try {
    if (!health.eligible.llm) {
      result.status = 'blocked';
      result.reason = health.llmBackend?.reason ?? 'LLM local configuration API unavailable; see drive-doctor.json';
      result.unverifiedEntryPoints = routes.map(route => route + ' → 模型提供商');
      return;
    }
    if (!writing) {
      for (const route of routes) {
        if (route !== '/') await step(`从 ${route} 打开模型配置`, () => goto(route));
        await step('读取四个普通字段与密钥配置状态', open);
        await capture('llm-loaded');
        await step('刷新真实LLM配置', async () => {
          const pending = watch('GET');
          await form.getByRole('button', { name: '刷新', exact: true }).click();
          const values = await readResponse(await pending, '刷新模型配置');
          await checkInputs(values);
          await form.getByRole('status').filter({ hasText: '已读取服务器模型配置' }).waitFor();
          assert.equal(await dialog.getByRole('button', { name: '保存', exact: true }).isDisabled(), true);
        });
        await capture('llm-refreshed');
        await step('关闭模型配置', close);
        result.entrypointsCovered.push(route + ' → 模型提供商 → 刷新');
      }
      return;
    }
    await step('记录LLM原始公开配置与密钥状态', async () => {
      original = await open();
      target = llmTarget(original);
      saveJson('original-config.json', { capturedAt: new Date().toISOString(), endpoint, values: original, target });
      result.original = original;
      result.target = target;
    });
    await capture('llm-original');
    await step('温度3由Rust拒绝且零POST', async () => {
      const attempts = writeGuard.attempts;
      await form.locator('#temperature').fill('3');
      await dialog.getByRole('button', { name: '保存', exact: true }).click();
      await form.getByRole('alert').filter({ hasText: /温度.*-2.*2/ }).waitFor();
      assert.equal(writeGuard.attempts, attempts, 'Invalid temperature must generate zero write requests');
    });
    await capture('llm-invalid-temperature');
    await step('仅修改max_tokens并保留原密钥', () => saveThroughUi(target, original, '保存LLM目标'));
    await capture('llm-saved');
    await step('重开GET确认LLM目标和密钥状态', async () => {
      await close();
      assert.ok(sameLlmConfig(await open(), target), 'LLM persisted values differ from the target');
    });
    await capture('llm-persisted');
    await step('通过UI恢复原值含null', () => saveThroughUi(original, target, '恢复LLM原值'));
    await capture('llm-restored');
    await step('重开GET确认LLM原值与密钥状态恢复', async () => {
      await close();
      assert.ok(sameLlmConfig(await open(), original), 'LLM original values or key status were not preserved');
      await close();
    });
    result.entrypointsCovered = ['/ → 模型提供商 → 保存 → 重开GET', '/ → 模型提供商 → UI恢复原值 → 重开GET'];
  } catch (error) {
    if (error.code === 'LLM_UNSAFE_MAX_TOKENS' && !writeAttempted) {
      result.status = 'blocked'; result.reason = error.message;
    } else throw error;
  } finally {
    writeGuard.expected = null;
    if (original && target && writeAttempted) {
      result.configCleanup = {};
      try {
        writeGuard.journal.finalize();
        await restoreConfigAfterWrites({ journal: writeGuard.journal, original, target, read: readCurrent,
          equals: sameLlmConfig, decide: llmRestoreDecision, progress: result.configCleanup,
          write: async values => {
            const payload = llmPayload(values);
            assert.ok(allowedConfigWrite({ feature: 'llm-save', allowConfigWrite, baseUrl: state.url,
              url: endpoint, method: 'POST', payload, expected: llmPayload(original) }), 'LLM restore payload must preserve the key');
            const entry = writeGuard.journal.begin({ url: endpoint, payload, source: 'compensation' });
            record({ action: '异常路径恢复LLM普通字段', phase: 'begin', payload });
            await withinPostDeadline(async () => {
              const response = await page.request.post(endpoint, { data: payload, timeout: 10000, maxRedirects: 0, maxRetries: 0 });
              if (response.status() >= 300 && response.status() < 400) throw new Error('LLM restore redirect blocked; server outcome unknown');
              writeGuard.journal.complete(entry, response.status());
              assert.ok(sameLlmConfig(await readResponse(response, 'LLM补偿恢复', 'POST'), original), 'LLM compensation response differs from original public fields');
            }, writeGuard.journal);
          } });
      } catch (error) {
        result.status = 'failed'; result.cleanupError = error.message;
        record({ action: 'LLM配置恢复需处理', error: error.message });
      }
    } else if (writing) result.configCleanup = { writeAttempted: false };
    if (writing) {
      if (writeGuard.journal.entries.some(entry => entry.outcome === 'outcome-unknown')) {
        result.status = 'failed'; result.configCleanup.decision = 'outcome-unknown'; delete result.configCleanup.restored;
      }
      saveJson('config-cleanup.json', { ...result.configCleanup, cleanupError: result.cleanupError });
    }
    saveJson('api-responses.json', responses);
  }
}
