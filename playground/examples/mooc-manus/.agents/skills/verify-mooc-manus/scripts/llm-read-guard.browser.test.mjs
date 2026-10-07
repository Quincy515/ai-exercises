#!/usr/bin/env node
// Optional Chrome regression with two isolated local HTTP fixtures. No app or
// configuration backend is started or contacted; the default suite stays pure.
import assert from 'node:assert/strict';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { driverInfo } from './flows.mjs';
import { excludeLlmDevStream, forwardLlmRead } from './llm-transport.mjs';
import { applyFinalSafetyGate, settleRouteHandlers, trackRouteHandler } from './route-lifecycle.mjs';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../..');
const listen = server => new Promise((resolve, reject) => {
  server.once('error', reject);
  server.listen(0, '127.0.0.1', resolve);
});
const close = server => new Promise(resolve => server.close(resolve));

test('LLM browser GET/HEAD preserve 200 and reject 302/307 before reaching another origin', async t => {
  const { api } = driverInfo(repo);
  const targetHits = [];
  const target = http.createServer((request, response) => {
    targetHits.push({ method: request.method, path: request.url });
    response.writeHead(200, { 'Access-Control-Allow-Origin': '*' });
    response.end('unexpected destination');
  });
  let browser, source;
  try {
    await listen(target);
    const targetUrl = `http://127.0.0.1:${target.address().port}/model-api`;
    source = http.createServer((request, response) => {
      const url = new URL(request.url, 'http://fixture.local');
      const status = url.searchParams.get('status');
      if (status === '302' || status === '307') {
        response.writeHead(Number(status), { Location: targetUrl }); response.end();
      } else if (status === 'drop') request.socket.destroy();
      else if (url.pathname === '/asset.js') {
        response.writeHead(200, { 'Content-Type': 'text/plain', 'X-Fixture-Source': 'local-source' });
        response.end('real source body');
      } else {
        response.writeHead(200, { 'Content-Type': 'text/html' });
        response.end('<!doctype html><title>LLM read guard fixture</title>');
      }
    });
    await listen(source);
    const baseUrl = `http://127.0.0.1:${source.address().port}`;
    browser = await api.chromium.launch({ channel: 'chrome', headless: true });
    for (const method of ['GET', 'HEAD']) for (const status of ['200', '302', '307', 'drop']) {
      await t.test(`${method} ${status}`, { timeout: 15000 }, async () => {
        const page = await browser.newPage(), blocked = [];
        try {
          await page.route('**/*', route => forwardLlmRead(route, blocked));
          await page.goto(baseUrl);
          const observed = await page.evaluate(async ({ method, status }) => {
            try {
              const response = await fetch('/asset.js?status=' + status + '&secret=fixture-only', { method });
              return { status: response.status, body: await response.text(), source: response.headers.get('X-Fixture-Source') };
            } catch { return { error: 'fetch rejected' }; }
          }, { method, status });
          if (status === '200') {
            assert.deepEqual(observed, { status: 200, body: method === 'GET' ? 'real source body' : '', source: 'local-source' });
            assert.deepEqual(blocked, []);
          } else {
            assert.equal(observed.error, 'fetch rejected');
            assert.equal(blocked.length, 1);
            assert.equal(blocked[0].method, method);
            assert.equal(blocked[0].url, baseUrl + '/asset.js');
            assert.equal(JSON.stringify(blocked).includes('fixture-only'), false);
            assert.equal(blocked[0].reason, status === 'drop' ? 'LLM read transport failed' : 'LLM read redirect blocked');
          }
          assert.deepEqual(targetHits, [], 'Redirect destination must receive zero requests');
          t.diagnostic(JSON.stringify({ method, scenario: status, observed, blocked, targetHits: targetHits.length }));
        } finally { await page.close(); }
      });
    }
  } finally {
    await browser?.close();
    if (source?.listening) await close(source);
    if (target.listening) await close(target);
  }
});

test('devtools exclusion and route teardown retain accurate browser evidence', async t => {
  const { api } = driverInfo(repo);
  let upstreamDevStreams = 0;
  const source = http.createServer((request, response) => {
    if (request.url.startsWith('/__tsd/console-pipe/sse')) upstreamDevStreams++;
    response.writeHead(200, { 'Content-Type': 'text/html' });
    response.end('<!doctype html><title>Route cleanup fixture</title>');
  });
  let browser;
  try {
    await listen(source);
    const baseUrl = `http://127.0.0.1:${source.address().port}`;
    browser = await api.chromium.launch({ channel: 'chrome', headless: true });
    await t.test('devtools stream is excluded before upstream and remains informational', async () => {
      const context = await browser.newContext(), page = await context.newPage();
      const pending = new Set();
      const result = { status: 'passed', blockedWrites: [], pageErrors: [], postOutcomes: [], ignoredDevRequests: [] };
      try {
        await page.route('**/*', route => trackRouteHandler(pending, async () => {
          if (await excludeLlmDevStream(route, baseUrl, result.ignoredDevRequests)) return;
          await forwardLlmRead(route, result.blockedWrites);
        }, error => result.pageErrors.push(error.message)));
        await page.goto(baseUrl);
        await page.evaluate(async () => { try { await fetch('/__tsd/console-pipe/sse'); } catch { /* Expected local abort. */ } });
      } finally { await context.close(); await settleRouteHandlers(pending); }
      applyFinalSafetyGate(result);
      assert.equal(upstreamDevStreams, 0);
      assert.equal(result.ignoredDevRequests.length, 1);
      assert.equal(result.ignoredDevRequests[0].reason, 'devtools stream excluded');
      assert.deepEqual(result.blockedWrites, []);
      assert.equal(result.status, 'passed');
      t.diagnostic(JSON.stringify({ scenario: 'devtools exclusion', upstreamDevStreams, status: result.status, ignored: result.ignoredDevRequests }));
    });
    await t.test('blocked transport arriving during context close still fails final evidence', async () => {
      const context = await browser.newContext(), page = await context.newPage();
      const pending = new Set();
      const result = { status: 'passed', blockedWrites: [], pageErrors: [], postOutcomes: [] };
      let entered, release;
      const started = new Promise(resolve => { entered = resolve; });
      const closing = new Promise(resolve => { release = resolve; });
      context.once('close', release);
      try {
        await page.route('**/*', route => trackRouteHandler(pending, async () => {
          if (new URL(route.request().url()).pathname === '/late-read') { entered(); await closing; }
          await forwardLlmRead(route, result.blockedWrites);
        }, error => result.pageErrors.push(error.message)));
        await page.goto(baseUrl);
        await page.evaluate(() => { void fetch('/late-read').catch(() => {}); });
        await started;
        assert.deepEqual(result.blockedWrites, [], 'Before close the old gate sees no failures');
      } finally { await context.close(); release(); await settleRouteHandlers(pending); }
      applyFinalSafetyGate(result);
      assert.equal(pending.size, 0);
      assert.equal(result.blockedWrites.length, 1);
      assert.equal(result.blockedWrites[0].reason, 'LLM read transport failed');
      assert.equal(result.status, 'failed');
      t.diagnostic(JSON.stringify({ scenario: 'late blocked read', status: result.status, blocked: result.blockedWrites }));
    });
  } finally {
    await browser?.close();
    if (source.listening) await close(source);
  }
});
