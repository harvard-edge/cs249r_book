import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { inspect } from 'node:util';
import vm from 'node:vm';
import { webcrypto } from 'node:crypto';
import { test } from 'node:test';
import ts from 'typescript';

// Exercise the public handler with isolated fetch, bindings, and console. No
// network or Cloudflare runtime is needed; TypeScript is already a dev dependency.
const source = readFileSync(new URL('../src/index.ts', import.meta.url), 'utf8');
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText;
const secret = 'PRIVATE_PROMPT_AND_API_KEY_DO_NOT_LOG';
const providers = [
  ['groq', 'GROQ_API_KEY'],
  ['openai', 'OPENAI_API_KEY'],
  ['anthropic', 'ANTHROPIC_API_KEY'],
  ['gemini', 'GEMINI_API_KEY'],
  ['openrouter', 'OPENROUTER_API_KEY'],
];

function runtime(fetch) {
  const logs = [];
  const exports = {};
  const context = vm.createContext({
    exports, fetch, Request, Response, AbortSignal, URL, TextEncoder, crypto: webcrypto,
    // Binding failures settle immediately; suppress the unused timeout in this
    // harness so it doesn't keep Node alive. These tests do not measure timing.
    setTimeout: () => 0,
    console: {
      warn: (...args) => logs.push(args),
      error: (...args) => logs.push(args),
    },
  });
  vm.runInContext(compiled, context);
  return { worker: exports.default, logs };
}

function request(route) {
  const body = route === '/ask' ? { question: secret } : {
    candidateMessage: secret,
    transcript: [],
    conductorContext: {
      track: 'ML', level: 'staff', duration: 30, timeExpired: false,
      currentQuestion: null, chainContext: null,
      coveredAreas: [], uncoveredAreas: [], areaRatings: {},
    },
  };
  return new Request(`https://worker.example${route}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

function env(extra) {
  return {
    RATE_LIMIT_KV: { get: async () => null, put: async () => {} },
    ...extra,
  };
}

for (const route of ['/ask', '/interview']) {
  for (const [provider, key] of providers) {
    test(`${route}: ${provider} error body stays private while HTTP status is logged`, async () => {
      const { worker, logs } = runtime(async () => new Response(secret, { status: 401 }));
      const response = await worker.fetch(request(route), env({ [key]: secret }));
      assert.equal(response.status, 502);
      const body = await response.text();
      assert.match(body, /all_providers_failed/);
      assert.doesNotMatch(body + inspect(logs), new RegExp(secret));
      assert.equal(logs.length, 2);
      assert.match(logs[0][0], new RegExp(provider));
      assert.equal(logs[0][1].code, 'http_error');
      assert.equal(logs[0][1].status, 401);
    });
  }

  for (const [name, upstream, code] of [
    ['transport error', async () => { throw new Error(secret); }, 'upstream_error'],
    ['invalid JSON', async () => new Response(secret), 'invalid_json'],
    ['invalid response shape', async () => Response.json({ choices: [{ message: { content: { secret } } }] }), 'upstream_error'],
    ['empty response', async () => Response.json({}), 'empty_response'],
  ]) {
    test(`${route}: ${name} logs only the worker diagnostic`, async () => {
      const { worker, logs } = runtime(upstream);
      const response = await worker.fetch(request(route), env({ OPENAI_API_KEY: secret }));
      assert.equal(response.status, 502);
      assert.doesNotMatch(await response.text() + inspect(logs), new RegExp(secret));
      assert.equal(logs[0][1].code, code);
      assert.deepEqual(Object.keys(logs[0][1]), ['code']);
    });
  }

  test(`${route}: binding error messages and metadata stay private`, async () => {
    const { worker, logs } = runtime(() => { throw new Error('unexpected fetch'); });
    const response = await worker.fetch(request(route), env({
      AI: { run: async () => { throw Object.assign(new Error(secret), { body: secret, apiKey: secret }); } },
    }));
    assert.equal(response.status, 502);
    assert.doesNotMatch(await response.text() + inspect(logs), new RegExp(secret));
    assert.equal(logs[0][1].code, 'upstream_error');
  });

  test(`${route}: provider failure still falls through in configured order`, async () => {
    const calls = [];
    const { worker, logs } = runtime(async (url) => {
      calls.push(url);
      return calls.length === 1
        ? new Response(secret, { status: 503 })
        : Response.json({ choices: [{ message: { content: 'Safe answer' } }] });
    });
    const response = await worker.fetch(request(route), env({
      GROQ_API_KEY: secret, OPENAI_API_KEY: secret, PROVIDER_PRIORITY: 'openai,groq',
    }));
    assert.equal(response.status, 200);
    assert.equal((await response.json()).provider, 'groq');
    assert.match(calls[0], /api\.openai\.com/);
    assert.match(calls[1], /api\.groq\.com/);
    assert.equal(logs.length, 1);
    assert.doesNotMatch(inspect(logs), new RegExp(secret));
  });
}
