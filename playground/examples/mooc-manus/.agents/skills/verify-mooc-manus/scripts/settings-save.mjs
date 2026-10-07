#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { allowedConfigWrite, configKeys, configPath, requireConfigWriteAuthorization, restoreConfigAfterWrites, sameConfig, validConfig, withinPostDeadline } from './config-write.mjs';

export async function driveSettingsSave({ page, state, health, result, step, capture, record, directory, writeGuard, allowConfigWrite }) {
  requireConfigWriteAuthorization(['settings-save'], allowConfigWrite);
  const endpoint = state.url + configPath;
  const saveJson = (name, value) => fs.writeFileSync(path.join(directory, name), JSON.stringify(value, null, 2) + '\n');
  const dialog = page.getByRole('dialog', { name: 'MoocManus 设置' });
  const responses = [];
  let original, target;
  let writeAttempted = false;
  result.unverifiedEntryPoints = ['/sessions/1 → 打开设置 → 保存', '/schedules → 打开设置 → 保存', '/library → 打开设置 → 保存'];
  result.unverified = ['Electron 保存', '移动端保存', '故障模拟与保存中关闭保护', '并发第三方写入的原子冲突检测'];
  const watch = method => {
    const pending = page.waitForResponse(response => response.url() === endpoint && response.request().method() === method);
    pending.catch(() => {});
    return pending;
  };
  const readResponse = async (response, action, method = 'GET') => {
    assert.equal(response.status(), 200, `${action} must return a real HTTP 200`);
    const values = await response.json();
    assert.ok(validConfig(values), `${action} returned values outside the supported write contract`);
    responses.push({ action, method: response.request?.().method() ?? method, status: response.status(), values });
    saveJson('api-responses.json', responses);
    return values;
  };
  const checkInputs = async expected => {
    for (const key of configKeys) {
      await page.waitForFunction(({ key, value }) => document.getElementById(key)?.value === String(value), { key, value: expected[key] });
      assert.equal(await dialog.locator('#' + key).inputValue(), String(expected[key]));
    }
  };
  const open = async () => {
    const response = watch('GET');
    await page.getByRole('button', { name: '打开设置', exact: true }).click();
    await dialog.getByRole('button', { name: '通用配置', exact: true }).click();
    const current = await readResponse(await response, '打开设置读取');
    await checkInputs(current);
    await dialog.getByRole('status').filter({ hasText: '已读取服务器配置' }).waitFor();
    return current;
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
  const saveThroughUi = async (expected, previous, label) => {
    const current = await readCurrent(`${label}前并发检查`);
    assert.ok(sameConfig(current, previous), `${label}: server changed since the last read; refusing to overwrite`);
    await dialog.locator('#max_iterations').fill(String(expected.max_iterations));
    writeGuard.expected = expected;
    const response = watch('POST');
    writeAttempted = true;
    try {
      await dialog.getByRole('button', { name: '保存', exact: true }).click();
      await withinPostDeadline(async () => {
        const received = await response;
        const failure = await received.finished();
        if (failure) throw new Error(String(failure));
        const entry = writeGuard.requests.get(received.request());
        assert.ok(entry, 'POST response must match an authorized request');
        writeGuard.journal.complete(entry, received.status());
        assert.deepEqual(await readResponse(received, label, 'POST'), expected);
      }, writeGuard.journal);
      await dialog.getByRole('status').filter({ hasText: '保存成功' }).waitFor();
      await checkInputs(expected);
      assert.equal(await dialog.getByRole('button', { name: '保存', exact: true }).isDisabled(), true);
    } finally { writeGuard.expected = null; }
  };
  try {
    if (!health.eligible.settings) {
      result.status = 'blocked';
      result.reason = 'Live Agent config backend is unavailable; settings-save performs no writes';
      result.unverifiedEntryPoints.unshift('/ → 打开设置 → 保存');
      await capture('backend-unavailable');
      return;
    }
    await step('打开设置并持久保存原始三个值', async () => {
      original = await open();
      target = { ...original, max_iterations: original.max_iterations === 999 ? 998 : original.max_iterations + 1 };
      saveJson('original-config.json', { capturedAt: new Date().toISOString(), endpoint, values: original, target });
      result.original = original;
      result.target = target;
      assert.equal(await dialog.getByRole('button', { name: '保存', exact: true }).isDisabled(), true);
    });
    await capture('original-values');
    await step('输入0提交由Rust校验拒绝且没有POST请求', async () => {
      const attemptedBefore = writeGuard.attempts;
      await dialog.locator('#max_iterations').fill('0');
      await dialog.getByRole('button', { name: '保存', exact: true }).click();
      await dialog.getByRole('alert').filter({ hasText: /最大迭代次数.*1.*999/ }).waitFor();
      assert.equal(writeGuard.attempts, attemptedBefore, 'Invalid input must be rejected before any API write');
      assert.equal(await dialog.locator('#max_iterations').inputValue(), '0');
    });
    await capture('invalid-input-rejected');
    await step('通过UI保存合法目标并收到真实200', () => saveThroughUi(target, original, '保存测试目标'));
    await capture('saved-target');
    await step('关闭重开由GET确认目标持久化', async () => {
      await close();
      assert.deepEqual(await open(), target);
    });
    await capture('target-persisted');
    await step('通过UI恢复原始配置', () => saveThroughUi(original, target, '恢复原始值'));
    await capture('restored-in-ui');
    await step('关闭重开由GET确认原始值已恢复', async () => {
      await close();
      assert.deepEqual(await open(), original);
      await close();
    });
    result.entrypointsCovered = ['/ → 打开设置 → 保存 → 关闭重开GET', '/ → 打开设置 → 恢复原值 → 关闭重开GET'];
  } finally {
    writeGuard.expected = null;
    if (original && target && writeAttempted) {
      try {
        writeGuard.journal.finalize();
        result.configCleanup = {};
        await restoreConfigAfterWrites({ journal: writeGuard.journal, original, target,
          read: readCurrent, progress: result.configCleanup, write: async values => {
            assert.ok(allowedConfigWrite({ feature: 'settings-save', allowConfigWrite, baseUrl: state.url,
              url: endpoint, method: 'POST', payload: values, expected: original }), 'Exceptional restore must pass the write policy');
            record({ action: '异常路径补偿恢复', phase: 'begin', method: 'POST', url: endpoint, values });
            // APIRequestContext bypasses page.route and uses the same policy and outcome journal.
            const entry = writeGuard.journal.begin({ url: endpoint, payload: values, source: 'compensation' });
            await withinPostDeadline(async () => {
              const response = await page.request.post(endpoint, { data: values, timeout: 10000, maxRedirects: 0 });
              writeGuard.journal.complete(entry, response.status());
              assert.deepEqual(await readResponse(response, '异常补偿恢复POST', 'POST'), original);
            }, writeGuard.journal);
            record({ action: '异常路径补偿恢复', phase: 'passed', method: 'POST', values });
          } });
      } catch (error) {
        result.status = 'failed';
        result.cleanupError = error.message;
        record({ action: '配置恢复需要处理', phase: 'failed', error: error.message });
      }
    } else {
      result.configCleanup = { writeAttempted: false, reason: 'No server write attempted' };
    }
    result.postOutcomes = writeGuard.journal.entries;
    if (result.postOutcomes.some(entry => entry.outcome === 'outcome-unknown')) {
      result.status = 'failed';
      result.configCleanup.decision = 'outcome-unknown';
      delete result.configCleanup.restored;
    }
    saveJson('config-cleanup.json', { ...result.configCleanup, cleanupError: result.cleanupError });
    saveJson('api-responses.json', responses);
  }
}
