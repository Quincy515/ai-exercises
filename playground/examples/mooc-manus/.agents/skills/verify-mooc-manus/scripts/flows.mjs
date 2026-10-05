#!/usr/bin/env node
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { createRequire } from 'node:module';

export function driverInfo(repo, { modulePath, channel = 'chrome' } = {}) {
  const require = createRequire(path.join(repo, 'apps/package.json'));
  const candidates = modulePath ? [modulePath] : [process.env.MOOC_PLAYWRIGHT_MODULE, 'playwright',
    path.join(os.homedir(), '.agents/skills/gstack/node_modules/playwright'),
    path.join(os.homedir(), '.codex/skills/gstack/node_modules/playwright')].filter(Boolean);
  for (const candidate of candidates) {
    try {
      const entry = require.resolve(candidate);
      const api = require(entry);
      if (!api.chromium) continue;
      const version = JSON.parse(fs.readFileSync(path.join(path.dirname(entry), 'package.json'))).version;
      return { api, entry, version, channel };
    } catch { /* Try the next existing installation; never install packages here. */ }
  }
  throw new Error('Playwright unavailable. Pass --playwright-module with an existing installation.');
}

const keys = ['max_iterations', 'max_retries', 'max_search_results'];
const files = ['go+java.pdf', '全家福.png', '2025年年中汇报.docx', '数据分析可视化看板.xsx', '数据看板动态演示.gif', 'ReActAgent.py'];

export async function driveFlows({ state, health, features, modulePath, channel }) {
  const driver = driverInfo(state.repo, { modulePath, channel });
  const report = { startedAt: new Date().toISOString(), head: state.source.head,
    workingTreeDigest: state.source.digest, surface: 'Web', url: state.url,
    driver: { entry: driver.entry, version: driver.version, channel: driver.channel },
    mocks: false, apiWriteGuard: true, features: [] };
  let browser;
  let currentFeature;
  try {
    browser = await driver.api.chromium.launch({ channel: driver.channel, headless: true });
    for (const feature of features) {
      currentFeature = feature;
      const directory = path.join(state.run, feature);
      fs.mkdirSync(directory);
      const context = await browser.newContext({ viewport: { width: 1280, height: 900 }, serviceWorkers: 'block' });
      await context.tracing.start({ screenshots: true, snapshots: true, sources: false });
      const page = await context.newPage();
      page.setDefaultTimeout(15000);
      const errors = [], network = [], blockedWrites = [];
      let sequence = 0;
      const result = { id: feature, status: 'passed', boundary: feature === 'settings' ? 'live API + Rust/WASM + UI' : 'built-in demo data + real UI interactions',
        entrypointsCovered: [], checks: [] };
      const record = item => fs.appendFileSync(path.join(directory, 'actions.jsonl'), JSON.stringify({ at: new Date().toISOString(), ...item }) + '\n');
      const step = async (action, fn) => {
        record({ action, phase: 'begin' });
        await fn();
        record({ action, phase: 'passed', url: page.url() });
        result.checks.push(action);
      };
      const capture = async label => {
        const prefix = `${String(++sequence).padStart(2, '0')}-${label}`;
        await page.screenshot({ path: path.join(directory, prefix + '.png'), animations: 'disabled' });
        fs.writeFileSync(path.join(directory, prefix + '.aria.txt'), await page.locator('body').ariaSnapshot());
        record({ evidence: prefix });
      };
      const button = name => page.getByRole('button', { name, exact: true });
      const watchConfig = () => {
        const pending = page.waitForResponse(r => r.url() === state.url + '/api/app_configs/agent' && r.request().method() === 'GET');
        // A click can fail before we await the response; keep that rejection handled.
        pending.catch(() => {});
        return pending;
      };
      const goto = async route => {
        await page.goto(state.url + route);
        await page.getByRole('navigation', { name: '功能导航' }).waitFor();
        assert.equal(await page.title(), 'Mooc Manus');
      };
      const openSession = async id => {
        const name = `打开会话 ${id}：图片合并为PDF的操作计划`;
        await button(name).click();
        await page.waitForURL(state.url + `/sessions/${id}`);
        await page.getByRole('region', { name: '会话任务详情' }).waitFor();
        assert.equal(await button(name).getAttribute('aria-current'), 'page');
      };
      page.on('pageerror', error => errors.push(error.message));
      page.on('response', response => {
        if (new URL(response.url()).pathname.startsWith('/api/')) network.push({ url: response.url(), status: response.status(), method: response.request().method() });
      });
      await page.route('**/api/**', async route => {
        if (!['GET', 'HEAD'].includes(route.request().method())) {
          blockedWrites.push({ method: route.request().method(), url: route.request().url() });
          await route.abort('blockedbyclient');
        } else await route.continue();
      });
      try {
        await step('打开首页', () => goto('/'));
        await capture('before');
        if (feature === 'settings') {
          const routes = ['/', '/sessions/1', '/schedules', '/library'];
          const responses = [];
          for (const route of routes) {
            if (route !== '/') await step(`从 ${route} 进入设置`, () => goto(route));
            const initialResponse = watchConfig();
            await step('点击打开设置', () => button('打开设置').click());
            const dialog = page.getByRole('dialog', { name: 'MoocManus 设置' });
            await dialog.getByRole('button', { name: '通用配置', exact: true }).click();
            const firstResponse = await initialResponse;
            if (!health.eligible.settings) {
              // Exercise the actual unavailable-backend path without fabricating a successful response.
              await dialog.getByRole('alert').waitFor();
              await capture('backend-unavailable');
              result.status = 'blocked';
              result.reason = 'localhost:5150 Agent config API is unavailable or failed contract checks; see drive-doctor.json';
              result.unverifiedEntryPoints = routes;
              break;
            }
            const check = async response => {
              assert.equal(response.status(), 200);
              const actual = await response.json();
              await page.waitForFunction(values => Object.entries(values).every(([key, value]) =>
                document.getElementById(key)?.value === String(value)), Object.fromEntries(keys.map(k => [k, actual[k]])));
              await dialog.getByRole('status').filter({ hasText: '已读取服务器配置' }).waitFor();
              responses.push({ route, status: response.status(), values: Object.fromEntries(keys.map(k => [k, actual[k]])) });
              for (const key of keys) {
                assert.equal(await dialog.locator('#' + key).inputValue(), String(actual[key]));
                assert.equal(await dialog.locator('#' + key).getAttribute('readonly'), '');
              }
              assert.equal(await dialog.getByRole('button', { name: '保存', exact: true }).isDisabled(), true);
            };
            await step('三个只读字段与真实 API 一致', () => check(firstResponse));
            await capture('settings-loaded');
            const responsePromise = watchConfig();
            await step('点击刷新并收到真实响应', async () => {
              await dialog.getByRole('button', { name: '刷新', exact: true }).click();
              await check(await responsePromise);
            });
            await capture('settings-refreshed');
            await step('关闭设置', () => dialog.getByRole('button', { name: '取消', exact: true }).click());
            await dialog.waitFor({ state: 'hidden' });
            result.entrypointsCovered.push(route + ' → 打开设置');
          }
          fs.writeFileSync(path.join(directory, 'api-responses.json'), JSON.stringify(responses, null, 2));
        }
        if (feature === 'sessions') {
          await step('侧栏进入会话 1', () => openSession(1));
          await capture('session-detail');
          await step('展开计划，确认三个步骤', async () => {
            assert.equal(await button('展开任务计划').getAttribute('aria-expanded'), 'false');
            await button('展开任务计划').click();
            assert.equal(await button('收起任务计划').getAttribute('aria-expanded'), 'true');
            assert.equal(await page.getByRole('list', { name: '任务步骤' }).getByRole('listitem').count(), 3);
          });
          await capture('plan-expanded');
          await step('收起计划', async () => {
            await button('收起任务计划').click();
            assert.equal(await page.getByRole('list', { name: '任务步骤' }).count(), 0);
          });
          await capture('plan-collapsed');
          await step('新聊天返回首页', async () => {
            await button('新聊天').click();
            await page.getByRole('region', { name: '新建会话任务' }).waitFor();
            assert.equal(new URL(page.url()).pathname, '/');
          });
          await step('会话 2 深链接与刷新', async () => {
            await goto('/sessions/2');
            await page.reload();
            await page.getByRole('region', { name: '会话任务详情' }).waitFor();
            assert.equal(await button('打开会话 2：图片合并为PDF的操作计划').getAttribute('aria-current'), 'page');
          });
          await capture('deep-link-reload');
          await step('首页导航与首页 Logo', async () => {
            await button('首页').click();
            await page.getByRole('region', { name: '新建会话任务' }).waitFor();
            await button('返回首页').click();
            assert.equal(new URL(page.url()).pathname, '/');
          });
          result.entrypointsCovered = ['侧栏会话入口', '/sessions/2 深链接+刷新', '新聊天', '首页导航', '首页 Logo'];
        }
        if (feature === 'files') {
          await step('从侧栏进入会话', () => openSession(1));
          await capture('before-dialog');
          await step('打开任务文件列表', () => button('查看会话文件').click());
          const dialog = page.getByRole('dialog', { name: '此任务中的所有文件' });
          await dialog.waitFor();
          const items = dialog.getByRole('list', { name: '任务文件' }).getByRole('listitem');
          await step('六个演示文件逐项可见', async () => {
            assert.equal(await items.count(), 6);
            for (const name of files) await dialog.getByText(name, { exact: true }).waitFor();
          });
          await capture('file-list');
          await step('关闭文件列表', async () => {
            await dialog.getByRole('button', { name: 'Close', exact: true }).click();
            await dialog.waitFor({ state: 'hidden' });
            await page.getByRole('region', { name: '会话任务详情' }).waitFor();
          });
          result.entrypointsCovered = ['会话 Header → 查看会话文件'];
          result.unimplemented = ['消息区查看全部', '文件下载', '文件预览'];
        }
        await capture('after');
        assert.deepEqual(blockedWrites, [], 'Unexpected API write; feature map may be outdated');
        assert.deepEqual(errors, [], 'Unexpected page script errors');
      } catch (error) {
        result.status = 'failed'; result.reason = error.stack;
        record({ phase: 'failed', error: error.message });
        try { await capture('failure'); } catch { /* Keep the other proof artifacts. */ }
      } finally {
        result.network = network; result.pageErrors = errors; result.blockedWrites = blockedWrites;
        try { await context.tracing.stop({ path: path.join(directory, 'trace.zip') }); }
        catch (error) { result.status = 'failed'; result.traceError = error.message; }
        try { await context.close(); }
        catch (error) { result.status = 'failed'; result.contextError = error.message; }
        fs.writeFileSync(path.join(directory, 'result.json'), JSON.stringify(result, null, 2) + '\n');
        report.features.push(result);
        fs.writeFileSync(path.join(state.run, 'results.json'), JSON.stringify(report, null, 2) + '\n');
      }
    }
  } catch (error) {
    report.infrastructureError = error.stack;
    for (const id of features) if (!report.features.some(item => item.id === id)) {
      report.features.push({ id, status: id === (currentFeature ?? features[0]) ? 'failed' : 'unverified', reason: 'Harness infrastructure failure' });
    }
    throw error;
  } finally {
    try { await browser?.close(); }
    finally {
      report.finishedAt = new Date().toISOString();
      fs.writeFileSync(path.join(state.run, 'results.json'), JSON.stringify(report, null, 2) + '\n');
    }
  }
  return report;
}
