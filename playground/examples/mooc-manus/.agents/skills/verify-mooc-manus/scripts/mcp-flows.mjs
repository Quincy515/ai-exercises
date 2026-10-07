#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { requireConfigWriteAuthorization, withinPostDeadline } from './config-write.mjs';
import { inspectNativeBackend } from './a2a-card-fixture.mjs';
import { startMcpFixture } from './mcp-http-fixture.mjs';
import { mcpFixtureFingerprint, mcpFixturePayload, mcpPath, mcpRecoveryStatus, readMcpList, readMcpWriteResponse, sameMcpList } from './mcp-policy.mjs';

export async function finalizeMcpFixture(fixture, result, requireToolList = false) {
  result.configCleanup ??= {};
  try {
    await fixture.close();
    result.configCleanup.fixtureStopped = true;
    const rejectedRequests = fixture.observations.filter(entry => !entry.accepted).length;
    const toolsListed = fixture.observations.some(entry => entry.operation === 'tools/list' && entry.accepted);
    result.fixtureSafetyGate = { checkedAfterClose: true, rejectedRequests, toolsListed };
    if (rejectedRequests || (requireToolList && !toolsListed)) {
      result.status = 'failed';
      result.cleanupError ??= 'MCP fixture protocol checks failed after shutdown; inspect safe request metadata';
    }
  } catch {
    result.status = 'failed';
    result.configCleanup.fixtureStopped = false;
    result.fixtureSafetyGate = { checkedAfterClose: false, closeFailed: true };
    result.cleanupError ??= 'MCP fixture shutdown failed; sensitive diagnostics excluded';
  }
}

export async function driveMcp({ feature, page, state, health, result, step, capture, directory, writeGuard, allowConfigWrite, goto }) {
  const writing = feature === 'mcp-write';
  if (writing) requireConfigWriteAuthorization(['mcp-write'], allowConfigWrite);
  const endpoint = state.url + mcpPath, routes = ['/', '/sessions/1', '/schedules', '/library'];
  const settings = page.getByRole('dialog', { name: 'MoocManus 设置', exact: true });
  const list = settings.getByRole('list', { name: 'MCP服务器列表', exact: true });
  const editor = page.getByRole('dialog', { name: '添加或更新 MCP 服务器', exact: true });
  const saveJson = (name, value) => fs.writeFileSync(path.join(directory, name), JSON.stringify(value, null, 2) + '\n');
  const responses = [];
  let fixture, baseline, current, ownedFingerprint = null, createAttempted = false, deleteConfirmed = false, revision = 'initial';
  result.proofBoundary = 'Public MCP list and controlled fixture only; secret configuration values never recorded';
  result.unverified = ['stdio 命令执行', '真实远程工具调用', '既有配置秘密字段一致性', '并发同名覆盖的原子冲突检测', 'Electron', '移动端'];
  if (writing) {
    result.externalBoundary = 'Local streamable HTTP protocol fixture; initialize and tools/list only, tools/call rejected';
    result.unverifiedEntryPoints = routes.slice(1).map(route => route + ' → MCP 服务器 → 写操作');
  }
  const watch = (method, url = endpoint) => {
    const pending = page.waitForResponse(response => response.url() === url && response.request().method() === method);
    pending.catch(() => {}); return pending;
  };
  const readResponse = async (response, action) => {
    assert.equal(response.status(), 200, 'MCP list request must return HTTP 200');
    current = readMcpList(await response.text());
    responses.push({ action, method: 'GET', status: 200, mcp_servers: current }); saveJson('api-responses.json', responses);
    return current;
  };
  const checkList = async rows => {
    await list.waitFor({ state: 'attached' });
    await page.waitForFunction(count => document.querySelectorAll('[aria-label="MCP服务器列表"] [data-mcp-name]').length === count, rows.length);
    for (const row of rows) {
      const index = await list.locator('[data-mcp-name]').evaluateAll((nodes, name) => nodes.findIndex(node => node.getAttribute('data-mcp-name') === name), row.server_name);
      assert.ok(index >= 0, 'Public MCP server name must be rendered');
      const item = list.locator('[data-mcp-name]').nth(index);
      await item.getByText(row.server_name, { exact: true }).waitFor();
      assert.equal(await item.getAttribute('data-mcp-transport'), row.transport);
      for (const tool of row.tools) await item.getByText(tool, { exact: true }).waitFor();
      const toggle = item.getByRole('switch', { name: '启用 ' + row.server_name, exact: true });
      await page.waitForFunction(({ name, enabled }) => Array.from(document.querySelectorAll('[data-mcp-name]'))
        .find(node => node.getAttribute('data-mcp-name') === name)?.querySelector('[role="switch"]')?.getAttribute('aria-checked') === String(enabled), { name: row.server_name, enabled: row.enabled });
      assert.equal(await toggle.isEnabled(), true);
    }
    assert.equal(await settings.getByRole('button', { name: '添加配置', exact: true }).isEnabled(), true);
  };
  const open = async () => {
    const pending = watch('GET');
    await page.getByRole('button', { name: '打开设置', exact: true }).click();
    await settings.getByRole('button', { name: 'MCP 服务器', exact: true }).click();
    const rows = await readResponse(await pending, '打开MCP列表');
    await settings.getByRole('status').filter({ hasText: '已读取 MCP 服务器列表' }).waitFor();
    await checkList(rows); return rows;
  };
  const readCurrent = async action => readResponse(await page.request.get(endpoint, { timeout: 10000, maxRedirects: 0, maxRetries: 0 }), action);
  const baselineUnchanged = rows => assert.ok(sameMcpList(rows.filter(row => row.server_name !== fixture.descriptor.serverName), baseline), 'Original MCP public metadata changed; stop without modifying it');
  const ownRow = () => list.locator(`[data-mcp-name="${fixture.descriptor.serverName}"]`);
  const requireOwned = rows => {
    assert.equal(ownedFingerprint, mcpFixtureFingerprint(fixture.descriptor), 'MCP ownership needs a confirmed fixture fingerprint');
    const own = rows.find(row => row.server_name === fixture.descriptor.serverName);
    assert.ok(own && own.transport === 'streamable_http', 'Owned MCP name or transport changed');
    if (own.enabled) assert.deepEqual(own.tools, [fixture.descriptor.toolName], 'Owned MCP tools differ from the controlled fixture');
    return own;
  };
  const mutate = async (action, click, { enabled = true, nextRevision = revision } = {}) => {
    const before = await readCurrent(action + '前检查'); baselineUnchanged(before);
    if (action === 'create') {
      assert.ok(!before.some(row => row.server_name === fixture.descriptor.serverName), 'Fixture name already exists; never overwrite it');
      const native = inspectNativeBackend(state.repo); saveJson('backend-process-before-create.json', native);
      assert.ok(native.ok && native.pid === health.mcpLocalBackend?.pid && native.processIdentity === health.mcpLocalBackend?.processIdentity, 'Native MCP backend identity changed');
    } else requireOwned(before);
    const url = ['create', 'update'].includes(action) ? endpoint : `${endpoint}/${encodeURIComponent(fixture.descriptor.serverName)}/${action}`;
    const expected = { action, fixture: fixture.descriptor, baseline, ownedFingerprint, createAttempted, enabled, revision: nextRevision };
    if (action === 'create') createAttempted = true;
    writeGuard.expected = expected;
    const pending = watch('POST', url), refreshed = watch('GET');
    try {
      await click();
      await withinPostDeadline(async () => {
        const response = await pending;
        if (await response.finished()) throw new Error('MCP write transport did not complete');
        const entry = writeGuard.requests.get(response.request()); assert.ok(entry, 'MCP response must match an authorized operation');
        writeGuard.journal.complete(entry, response.status());
        assert.equal(response.status(), 200, 'MCP write request must return HTTP 200');
        const safe = readMcpWriteResponse(await response.text(), { fixture: fixture.descriptor, action, enabled, revision: nextRevision });
        responses.push({ action, method: 'POST', status: 200, mcp_servers: safe.mcp_servers }); saveJson('api-responses.json', responses);
        if (action === 'create') {
          ownedFingerprint = safe.ownedFingerprint;
          saveJson('owned-record.json', { server_name: fixture.descriptor.serverName, transport: 'streamable_http', fingerprint: ownedFingerprint });
        }
      }, writeGuard.journal);
      revision = nextRevision;
      const rows = await readResponse(await refreshed, action + '后自动GET');
      await checkList(rows); baselineUnchanged(rows); return rows;
    } finally { writeGuard.expected = null; }
  };
  try {
    if (!health.eligible.mcp || (writing && !health.mcpLocalBackend?.ok)) {
      result.status = 'blocked'; result.reason = writing && !health.mcpLocalBackend?.ok ? health.mcpLocalBackend?.reason : 'MCP list API unavailable or public contract mismatch';
      result.unverifiedEntryPoints = routes.map(route => route + ' → MCP 服务器'); return;
    }
    if (!writing) {
      for (const route of routes) {
        if (route !== '/') await step('从 ' + route + ' 进入MCP设置', () => goto(route));
        await step('读取并核对MCP公开列表', open); await capture('mcp-loaded');
        await step('刷新MCP列表', async () => {
          const pending = watch('GET'); await settings.getByRole('button', { name: '刷新', exact: true }).click();
          await checkList(await readResponse(await pending, '刷新MCP列表'));
        });
        await capture('mcp-refreshed');
        await settings.getByRole('button', { name: '关闭', exact: true }).click(); await settings.waitFor({ state: 'hidden' });
        result.entrypointsCovered.push(route + ' → MCP 服务器 → 刷新');
      }
      return;
    }
    await step('记录MCP公开基线与启动受控HTTP夹具', async () => {
      baseline = await open(); saveJson('baseline-visible-list.json', { mcp_servers: baseline });
      fixture = await startMcpFixture(directory);
      result.fixture = fixture.descriptor;
      const quoted = "'" + path.join(directory, 'fixture.json').replaceAll("'", "'\\''") + "'";
      result.recoveryCommand = `node .agents/skills/verify-mooc-manus/scripts/mcp-http-fixture.mjs serve --fixture ${quoted}`;
      assert.ok(!baseline.some(row => row.server_name === fixture.descriptor.serverName), 'Random fixture name collision; abort before create');
    });
    await capture('mcp-baseline');
    await step('无效JSON由Rust拒绝且零POST', async () => {
      await settings.getByRole('button', { name: '添加配置', exact: true }).click();
      await editor.getByLabel('MCP服务器配置', { exact: true }).fill('{broken');
      const before = writeGuard.attempts; await editor.getByRole('button', { name: '保存配置', exact: true }).click();
      await editor.getByRole('alert').waitFor(); assert.equal(writeGuard.attempts, before, 'Invalid MCP JSON must generate zero POSTs');
    });
    await capture('mcp-invalid-json');
    await step('新增唯一MCP测试配置并确认工具', async () => {
      await editor.getByLabel('MCP服务器配置', { exact: true }).fill(JSON.stringify(mcpFixturePayload(fixture.descriptor)));
      requireOwned(await mutate('create', () => editor.getByRole('button', { name: '保存配置', exact: true }).click()));
      await editor.waitFor({ state: 'hidden' });
    });
    await capture('mcp-created');
    await step('同名更新仅修改本次测试配置', async () => {
      await settings.getByRole('button', { name: '添加配置', exact: true }).click();
      await editor.getByLabel('MCP服务器配置', { exact: true }).fill(JSON.stringify(mcpFixturePayload(fixture.descriptor, 'updated')));
      requireOwned(await mutate('update', () => editor.getByRole('button', { name: '保存配置', exact: true }).click(), { nextRevision: 'updated' }));
      await editor.waitFor({ state: 'hidden' });
    });
    await capture('mcp-updated');
    for (const enabled of [false, true]) {
      await step(enabled ? '启用本次MCP测试项' : '禁用本次MCP测试项', async () => {
        const rows = await mutate('enabled', () => ownRow().getByRole('switch', { name: '启用 ' + fixture.descriptor.serverName, exact: true }).click(), { enabled });
        assert.equal(requireOwned(rows).enabled, enabled);
      });
      await capture(enabled ? 'mcp-enabled' : 'mcp-disabled');
    }
    await step('删除本次MCP测试项并独立GET确认', async () => {
      await ownRow().getByRole('button', { name: '删除 ' + fixture.descriptor.serverName, exact: true }).click();
      const confirmation = page.getByRole('dialog', { name: '删除 MCP 服务器', exact: true });
      const rows = await mutate('delete', () => confirmation.getByRole('button', { name: '确认删除', exact: true }).click());
      assert.ok(sameMcpList(rows, baseline), 'MCP visible list not restored after delete');
      assert.ok(sameMcpList(await readCurrent('删除后独立GET'), baseline), 'MCP final public list differs from baseline');
      deleteConfirmed = true; await confirmation.waitFor({ state: 'hidden' });
    });
    await capture('mcp-deleted');
    await settings.getByRole('button', { name: '关闭', exact: true }).click();
    result.entrypointsCovered = ['/ → MCP 服务器 → Create → Same-name update → Disable → Enable → Delete → GET'];
  } finally {
    writeGuard.expected = null;
    if (fixture) {
      try {
        writeGuard.journal.finalize();
        if (createAttempted && !deleteConfirmed) { try { await readCurrent('异常路径只读定位MCP测试名'); } catch { current = null; } }
        result.configCleanup = mcpRecoveryStatus({ journal: writeGuard.journal, baseline, current, fixture: fixture.descriptor, ownedFingerprint, createAttempted, deleteConfirmed });
        if (result.configCleanup.manualCleanupRequired) {
          result.status = 'failed'; result.cleanupError = 'MCP operation requires supervised reconciliation; preserve the fixture, prove its configuration fingerprint, and remove only the owned name. Never repeat Create.';
        }
      } catch {
        result.status = 'failed'; result.cleanupError = 'MCP cleanup needs manual review; sensitive diagnostics excluded';
        result.configCleanup = { manualCleanupRequired: createAttempted, ownedName: ownedFingerprint ? fixture.descriptor.serverName : null, decision: 'manual-reconciliation-required' };
      } finally {
        await finalizeMcpFixture(fixture, result, deleteConfirmed);
        saveJson('config-cleanup.json', { ...result.configCleanup, cleanupError: result.cleanupError });
      }
    }
    saveJson('api-responses.json', responses);
  }
}
