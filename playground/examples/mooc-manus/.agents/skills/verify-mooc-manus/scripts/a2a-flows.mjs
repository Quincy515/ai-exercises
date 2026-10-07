#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { requireConfigWriteAuthorization, withinPostDeadline } from './config-write.mjs';
import { a2aPath, a2aRecoveryStatus, createA2aOwnership, readA2aList, sameVisibleA2a } from './a2a-policy.mjs';
import { inspectNativeBackend, startCardFixture } from './a2a-card-fixture.mjs';

export async function driveA2a({ feature, page, state, health, result, step, capture, record, directory, writeGuard, allowConfigWrite, goto }) {
  const writing = feature === 'a2a-write';
  if (writing) requireConfigWriteAuthorization(['a2a-write'], allowConfigWrite);
  const endpoint = state.url + a2aPath;
  const settings = page.getByRole('dialog', { name: 'MoocManus 设置', exact: true });
  const list = settings.getByRole('list', { name: '远程Agent列表', exact: true });
  const routes = ['/', '/sessions/1', '/schedules', '/library'];
  const saveJson = (name, value) => fs.writeFileSync(path.join(directory, name), JSON.stringify(value, null, 2) + '\n');
  const responses = [];
  const shellQuote = value => "'" + value.replaceAll("'", "'\\''") + "'";
  let baseline, current, fixture, owned, ownership, createAttempted = false, deleteConfirmed = false;
  result.proofBoundary = 'Visible Agent Card list only; no claim of full database configuration equality';
  result.unverified = ['真实远程Agent调用', '不可见的Card加载失败记录', '完整数据库配置一致性', 'Electron', '移动端'];
  if (writing) {
    result.externalBoundary = 'local controlled Agent Card; backend configuration HTTP is real; all invocation rejected';
    result.unverifiedEntryPoints = routes.slice(1).map(route => route + ' → A2A Agent配置 → 写操作');
  }
  const watch = (method, url = endpoint) => {
    const pending = page.waitForResponse(response => response.url() === url && response.request().method() === method);
    pending.catch(() => {}); return pending;
  };
  const readResponse = async (response, action) => {
    assert.equal(response.status(), 200, action + ' must return HTTP 200');
    current = readA2aList(await response.text());
    responses.push({ action, method: 'GET', status: 200, a2a_servers: current });
    saveJson('api-responses.json', responses);
    return current;
  };
  const checkList = async rows => {
    await list.waitFor({ state: 'attached' });
    await page.waitForFunction(() => Array.from(document.querySelectorAll('button')).some(button => button.textContent.trim() === '新增远程Agent' && !button.disabled));
    await page.waitForFunction(count => document.querySelectorAll('[aria-label="远程Agent列表"] [data-a2a-id]').length === count, rows.length);
    if (!rows.length) await settings.getByText('暂无可展示的远程 Agent。', { exact: false }).waitFor();
    for (const row of rows) {
      // IDs are read from DOM, rather than interpolated into selectors.
      const index = await list.locator('[data-a2a-id]').evaluateAll((elements, id) => elements.findIndex(element => element.getAttribute('data-a2a-id') === id), row.id);
      assert.ok(index >= 0, 'Visible A2A ID must be rendered');
      const actual = list.locator('[data-a2a-id]').nth(index);
      await actual.getByText(row.name || '未命名 Agent', { exact: true }).waitFor();
      await actual.getByText(row.description || '此 Agent 暂未提供描述。', { exact: true }).waitFor();
      await actual.getByText('输入: ' + (row.input_modes.join(', ') || '未提供'), { exact: true }).waitFor();
      await actual.getByText('输出: ' + (row.output_modes.join(', ') || '未提供'), { exact: true }).waitFor();
      assert.equal(await actual.getByText('流式输出', { exact: true }).count(), row.streaming ? 1 : 0);
      assert.equal(await actual.getByText('推送通知', { exact: true }).count(), row.push_notifications ? 1 : 0);
      const toggle = actual.getByRole('switch', { name: `启用 ${row.name || '未命名 Agent'}`, exact: true });
      await page.waitForFunction(({ id, checked }) => Array.from(document.querySelectorAll('[data-a2a-id]'))
        .find(element => element.getAttribute('data-a2a-id') === id)?.querySelector('[role="switch"]')?.getAttribute('aria-checked') === String(checked), { id: row.id, checked: row.enabled });
      assert.equal(await toggle.getAttribute('aria-checked'), String(row.enabled));
      assert.equal(await toggle.isEnabled(), true);
    }
    assert.equal(await settings.getByRole('button', { name: '新增远程Agent', exact: true }).isEnabled(), true);
  };
  const open = async () => {
    const pending = watch('GET');
    await page.getByRole('button', { name: '打开设置', exact: true }).click();
    await settings.getByRole('button', { name: 'A2A Agent配置', exact: true }).click();
    const rows = await readResponse(await pending, '打开A2A列表');
    await settings.getByRole('status').filter({ hasText: '已读取远程 Agent 列表' }).waitFor();
    await checkList(rows); return rows;
  };
  const readCurrent = async action => {
    const response = await page.request.get(endpoint, { timeout: 10000, maxRedirects: 0, maxRetries: 0 });
    return readResponse(response, action);
  };
  const checkBaseline = rows => assert.ok(sameVisibleA2a(rows.filter(row => row.id !== owned?.id), baseline), 'Previously visible A2A records changed; stop without touching them');
  const locateOwned = rows => {
    try { return ownership.find(rows); }
    catch (error) {
      if (error.code === 'A2A_OWNERSHIP_CONFLICT') saveJson('ownership-conflict.json', { knownId: ownership.id, candidate: error.candidate });
      throw error;
    }
  };
  const ownedRow = () => list.locator(`[data-a2a-id="${owned.id}"]`);
  const mutate = async (action, click, enabled) => {
    const before = await readCurrent(action + '前检查');
    if (action !== 'create') {
      const matched = locateOwned(before);
      assert.ok(matched && matched.id === owned.id, 'Owned A2A record must still be uniquely visible');
      owned = matched;
    }
    checkBaseline(before);
    if (action === 'create') {
      const local = inspectNativeBackend(state.repo);
      saveJson('backend-process-before-create.json', local);
      assert.ok(local.ok && local.pid === health.a2aLocalBackend?.pid && local.processIdentity === health.a2aLocalBackend?.processIdentity,
        'Native backend identity changed; creating the loopback fixture configuration is blocked');
    }
    const url = action === 'create' ? endpoint : `${endpoint}/${owned.id}/${action}`;
    const expected = { action, fixture: fixture.descriptor, baseline, owned, knownId: ownership.id, enabled, createAttempted };
    if (action === 'create') createAttempted = true; // Never repeat Create, even when its outcome is uncertain.
    writeGuard.expected = expected;
    const response = watch('POST', url), refreshed = watch('GET');
    try {
      await click();
      await withinPostDeadline(async () => {
        const received = await response;
        const failure = await received.finished();
        if (failure) throw new Error('A2A POST response did not complete');
        const entry = writeGuard.requests.get(received.request());
        assert.ok(entry, 'A2A response must belong to the authorized operation');
        writeGuard.journal.complete(entry, received.status());
        assert.equal(received.status(), 200, 'A2A operation must return HTTP 200');
        assert.equal((await received.text()).trim(), 'null', 'A2A write response must be JSON null');
        responses.push({ action, method: 'POST', url, status: 200, body: null });
        saveJson('api-responses.json', responses);
      }, writeGuard.journal);
      const rows = await readResponse(await refreshed, action + '后UI自动GET');
      await checkList(rows);
      return rows;
    } finally { writeGuard.expected = null; }
  };
  try {
    if (!health.eligible.a2a || (writing && !health.a2aLocalBackend?.ok)) {
      result.status = 'blocked'; result.reason = writing && !health.a2aLocalBackend?.ok
        ? health.a2aLocalBackend?.reason : 'A2A list API unavailable or contract mismatch';
      result.unverifiedEntryPoints = routes.map(route => route + ' → A2A Agent配置'); return;
    }
    if (!writing) {
      for (const route of routes) {
        if (route !== '/') await step(`从 ${route} 进入A2A配置`, () => goto(route));
        await step('读取并核对真实A2A列表', open);
        await capture('a2a-loaded');
        await step('刷新A2A列表', async () => {
          const pending = watch('GET');
          await settings.getByRole('button', { name: '刷新', exact: true }).click();
          await checkList(await readResponse(await pending, '刷新A2A列表'));
        });
        await capture('a2a-refreshed');
        await step('关闭A2A设置', async () => {
          await settings.getByRole('button', { name: '关闭', exact: true }).click();
          await settings.waitFor({ state: 'hidden' });
        });
        result.entrypointsCovered.push(route + ' → A2A Agent配置 → 刷新');
      }
      return;
    }
    await step('读取原始可见列表并启动本地Card夹具', async () => {
      baseline = await open();
      saveJson('baseline-visible-list.json', { a2a_servers: baseline, boundary: result.proofBoundary });
      fixture = await startCardFixture(directory);
      ownership = createA2aOwnership(baseline, fixture.descriptor, row => saveJson('owned-record.json', row));
      const reachable = await fetch(fixture.descriptor.baseUrl + '/.well-known/agent-card.json', { redirect: 'error', signal: AbortSignal.timeout(3000) });
      assert.equal(reachable.status, 200);
      assert.deepEqual(await reachable.json(), fixture.descriptor.card);
      result.fixture = fixture.descriptor;
      result.recoveryCommand = `node .agents/skills/verify-mooc-manus/scripts/a2a-card-fixture.mjs serve --fixture ${shellQuote(path.join(directory, 'fixture.json'))}`;
    });
    await capture('a2a-baseline');
    await step('无效地址由Rust拒绝且零POST', async () => {
      await settings.getByRole('button', { name: '新增远程Agent', exact: true }).click();
      const add = page.getByRole('dialog', { name: '添加远程Agent', exact: true });
      const count = writeGuard.attempts;
      await add.locator('#a2a_base_url').fill('not-an-http-url');
      await add.getByRole('button', { name: '添加', exact: true }).click();
      await add.getByRole('alert').filter({ hasText: /HTTP.*HTTPS/ }).waitFor();
      assert.equal(writeGuard.attempts, count);
    });
    await capture('a2a-invalid-url');
    await step('仅新增本次唯一Card配置', async () => {
      const add = page.getByRole('dialog', { name: '添加远程Agent', exact: true });
      await add.locator('#a2a_base_url').fill(fixture.descriptor.baseUrl);
      const rows = await mutate('create', () => add.getByRole('button', { name: '添加', exact: true }).click());
      owned = locateOwned(rows);
      assert.ok(owned, 'Created A2A record is not visible; preserve fixture evidence and never repeat Create');
      assert.equal(owned.enabled, true);
      checkBaseline(rows);
      await add.waitFor({ state: 'hidden' });
    });
    await capture('a2a-created');
    for (const enabled of [false, true]) {
      await step(enabled ? '启用本次测试Agent' : '禁用本次测试Agent', async () => {
        const rows = await mutate('enabled', () => ownedRow().getByRole('switch', { name: `启用 ${owned.name}`, exact: true }).click(), enabled);
        owned = locateOwned(rows);
        assert.ok(owned && owned.enabled === enabled, 'Owned Agent enabled state must match the operation');
        checkBaseline(rows);
      });
      await capture(enabled ? 'a2a-enabled' : 'a2a-disabled');
    }
    await step('仅删除明确属于本次的Agent', async () => {
      await ownedRow().getByRole('button', { name: `删除 ${owned.name}`, exact: true }).click();
      const confirmation = page.getByRole('dialog', { name: '删除远程Agent', exact: true });
      const rows = await mutate('delete', () => confirmation.getByRole('button', { name: '确认删除', exact: true }).click());
      assert.equal(locateOwned(rows), null);
      assert.ok(sameVisibleA2a(rows, baseline), 'Final visible A2A list must match its baseline');
      assert.equal(locateOwned(await readCurrent('删除后独立GET确认')), null);
      assert.ok(sameVisibleA2a(current, baseline));
      deleteConfirmed = true;
      await confirmation.waitFor({ state: 'hidden' });
      for (const key of ['Tab', 'Tab', 'Tab', 'Shift+Tab', 'Shift+Tab', 'Shift+Tab']) {
        await page.keyboard.press(key);
        // Base UI focus guards wrap Tab on requestAnimationFrame; wait for that settled focus.
        await page.waitForFunction(() => document.querySelector('[aria-label="远程Agent列表"]')
          ?.closest('[role="dialog"]')?.contains(document.activeElement));
        assert.ok(await settings.evaluate(element => element.contains(document.activeElement)), 'Focus must remain in parent settings after deleting a row');
      }
    });
    await capture('a2a-deleted');
    await settings.getByRole('button', { name: '关闭', exact: true }).click();
    result.entrypointsCovered = ['/ → A2A Agent配置 → Create → Disable → Enable → Delete → GET'];
  } finally {
    writeGuard.expected = null;
    if (fixture) {
      try {
        writeGuard.journal.finalize();
        if (createAttempted && !deleteConfirmed) {
          // Read-only reconciliation while the card is reachable. Never retry a write here.
          for (let attempt = 0; attempt < 3; attempt++) {
            try { await readCurrent('异常路径只读定位测试项'); }
            catch { current = null; continue; }
            // Identity conflicts must reach the recovery error handler, never be treated as a read retry.
            const found = locateOwned(current);
            if (found) { owned = found; break; }
          }
        }
        result.configCleanup = a2aRecoveryStatus({ journal: writeGuard.journal, baseline, current,
          fixture: fixture.descriptor, knownId: ownership?.id, createAttempted, deleteConfirmed });
        if (result.configCleanup.manualCleanupRequired) {
          result.status = 'failed'; result.cleanupError = 'A2A record requires supervised recovery: restart the saved fixture, use GET to prove ownership, then remove only the owned ID. Never repeat Create.';
        }
        assert.equal(fixture.observations.some(item => item.method !== 'GET' || !item.servedCard), false, 'Fixture received a prohibited invocation or unexpected request');
      } catch (error) {
        result.status = 'failed'; result.cleanupError = error.message;
        result.configCleanup = { manualCleanupRequired: createAttempted, ownedId: ownership?.id ?? null, decision: 'manual-reconciliation-required' };
      } finally {
        await fixture.close();
        result.configCleanup.fixtureStopped = true;
        saveJson('config-cleanup.json', { ...result.configCleanup, cleanupError: result.cleanupError, recoveryCommand: result.recoveryCommand });
      }
    }
    saveJson('api-responses.json', responses);
  }
}
