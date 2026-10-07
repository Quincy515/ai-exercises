#!/usr/bin/env node

// LLM reads include module/static asset requests. A browser redirect on any of
// them could escape the same-origin guard, so forward real responses explicitly.
export async function forwardLlmRead(route, blocked) {
  const request = route.request();
  const requested = new URL(request.url());
  const safe = { method: request.method(), url: requested.origin + requested.pathname };
  const reject = async (reason, status) => {
    blocked.push({ ...safe, reason, ...(status === undefined ? {} : { status }) });
    try { await route.abort('blockedbyclient'); } catch { /* The request may already have closed. */ }
  };
  if (!['GET', 'HEAD'].includes(request.method())) {
    await reject('LLM read transport accepts only GET/HEAD');
    return;
  }
  try {
    const response = await route.fetch({ maxRedirects: 0, maxRetries: 0, timeout: 10000 });
    if (response.status() >= 300 && response.status() < 400) {
      await reject('LLM read redirect blocked', response.status());
      return;
    }
    await route.fulfill({ response });
  } catch {
    await reject('LLM read transport failed');
  }
}

// This devtools-only SSE has no product role and never finishes its body.
// Exclude exactly this local GET in verification, without forwarding upstream.
export async function excludeLlmDevStream(route, baseUrl, ignored) {
  const request = route.request();
  const url = new URL(request.url());
  if (request.method() !== 'GET' || url.origin !== new URL(baseUrl).origin
    || url.pathname !== '/__tsd/console-pipe/sse') return false;
  ignored.push({ method: 'GET', url: url.origin + url.pathname, reason: 'devtools stream excluded' });
  try { await route.abort('blockedbyclient'); } catch { /* The context may already be closing. */ }
  return true;
}
