#!/usr/bin/env node
// Optional browser regression: local in-memory HTTP fixtures only. The normal
// verify.test.mjs suite remains browser-independent and never touches port 5150.
import assert from 'node:assert/strict';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { allowedConfigWrite, ConfigWriteJournal, forwardAuthorizedConfigWrite } from './config-write.mjs';
import { driverInfo } from './flows.mjs';

const payload = { max_iterations: 101, max_retries: 3, max_search_results: 10 };
const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../..');

test('real browser write transport preserves 200 and blocks redirects and network failures', async t => {
  const { api } = driverInfo(repo);
  let mode = '200';
  const received = [];
  const server = http.createServer(async (request, response) => {
    let body = '';
    for await (const chunk of request) body += chunk;
    received.push({ mode, path: request.url, method: request.method, body });
    if (request.url === '/api/app_configs/agent') {
      if (mode === '307' || mode === '308') {
        response.writeHead(Number(mode), { Location: '/redirect-target' });
        response.end();
      } else if (mode === 'drop') request.socket.destroy();
      else {
        response.writeHead(200, { 'Content-Type': 'application/json', 'X-Fixture-Source': 'real-local-http' });
        response.end(JSON.stringify(payload));
      }
    } else if (request.url === '/redirect-target') {
      response.writeHead(200);
      response.end('unauthorized redirect reached');
    } else {
      response.writeHead(200, { 'Content-Type': 'text/html' });
      response.end('<!doctype html><title>Write guard fixture</title>');
    }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const baseUrl = `http://127.0.0.1:${server.address().port}`;
  let browser;
  try {
    browser = await api.chromium.launch({ channel: 'chrome', headless: true });
    for (const scenario of ['200', '307', '308', 'drop']) {
      await t.test(scenario, { timeout: 15000 }, async () => {
        mode = scenario;
        const page = await browser.newPage();
        try {
          let settled;
          const terminal = new Promise(resolve => { settled = resolve; });
          const journal = new ConfigWriteJournal(entries => {
            if (entries.some(entry => entry.outcome !== 'pending')) settled();
          });
          await page.route('**/api/**', async route => {
            const request = route.request();
            const values = request.postDataJSON();
            assert.ok(allowedConfigWrite({ feature: 'settings-save', allowConfigWrite: true, baseUrl,
              url: request.url(), method: request.method(), payload: values, expected: payload }));
            const entry = journal.begin({ url: request.url(), payload: values });
            await forwardAuthorizedConfigWrite(route, journal, entry);
          });
          await page.goto(baseUrl);
          const observed = await page.evaluate(async values => {
            try {
              const response = await fetch('/api/app_configs/agent', { method: 'POST',
                headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(values) });
              return { status: response.status, body: await response.json(), source: response.headers.get('X-Fixture-Source') };
            } catch (error) { return { error: error.message }; }
          }, payload);
          await terminal;
          const posts = received.filter(item => item.mode === scenario && item.method === 'POST');
          assert.equal(posts.length, 1, 'Exactly one request reaches the approved endpoint');
          assert.equal(posts[0].path, '/api/app_configs/agent');
          assert.deepEqual(JSON.parse(posts[0].body), payload);
          assert.equal(received.some(item => item.path === '/redirect-target'), false);
          if (scenario === '200') {
            assert.deepEqual(observed, { status: 200, body: payload, source: 'real-local-http' });
            assert.equal(journal.entries[0].outcome, 'response-received');
          } else {
            assert.match(observed.error, /fetch/i);
            assert.equal(journal.entries[0].outcome, 'outcome-unknown');
          }
          t.diagnostic(JSON.stringify({ scenario, observed, posts, outcomes: journal.entries }));
        } finally { await page.close(); }
      });
    }
  } finally {
    await browser?.close();
    await new Promise(resolve => server.close(resolve));
  }
});
