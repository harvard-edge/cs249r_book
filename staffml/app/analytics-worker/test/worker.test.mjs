import assert from 'node:assert/strict';
import test from 'node:test';
import worker from '../worker.js';

function environment(overrides = {}) {
  const events = new Map();
  return {
    ADMIN_SECRET: 'test-operator-secret',
    STAFFML_ANALYTICS: {
      async get(key, options) {
        const value = events.get(key);
        return options?.type === 'json' && value ? JSON.parse(value) : value ?? null;
      },
      async put(key, value) { events.set(key, value); },
      async list({ prefix }) {
        return { keys: [...events.keys()].filter(key => key.startsWith(prefix)).map(name => ({ name })) };
      },
    },
    ...overrides,
  };
}

test('aggregate reads require a configured, nonempty admin secret', async () => {
  for (const secret of [undefined, null, '', '   ']) {
    for (const authorization of [undefined, 'Bearer undefined', 'Bearer null', 'Bearer', 'Bearer test-operator-secret']) {
      const headers = authorization ? { Authorization: authorization } : {};
      const response = await worker.fetch(new Request('https://analytics.test/', { headers }), environment({ ADMIN_SECRET: secret }));
      assert.equal(response.status, 401, `secret=${JSON.stringify(secret)}, authorization=${authorization}`);
    }
  }
});

test('aggregate reads reject absent and incorrect credentials and accept the configured bearer token', async () => {
  const env = environment();
  for (const authorization of [undefined, 'Bearer incorrect', 'Basic test-operator-secret']) {
    const headers = authorization ? { Authorization: authorization } : {};
    const response = await worker.fetch(new Request('https://analytics.test/summary', { headers }), env);
    assert.equal(response.status, 401);
  }
  const response = await worker.fetch(new Request('https://mlsysbook.ai/api/staffml-analytics', {
    headers: { Authorization: 'Bearer test-operator-secret' },
  }), env);
  assert.equal(response.status, 200);
  assert.ok((await response.json()).last7Days);
});

test('CORS permits each supported exact origin and normalizes the default HTTPS port', async () => {
  for (const origin of ['https://mlsysbook.ai', 'https://harvard-edge.github.io', 'http://localhost:3000', 'http://localhost:3001', 'https://mlsysbook.ai:443']) {
    const response = await worker.fetch(new Request('https://analytics.test/api/staffml-analytics', {
      method: 'OPTIONS', headers: { Origin: origin },
    }), environment());
    assert.equal(response.status, 204, origin);
    assert.equal(response.headers.get('Access-Control-Allow-Origin'), new URL(origin).origin);
  }
});

test('CORS rejects lookalike hosts, credentials, malformed origins, and unconfigured ports', async () => {
  for (const origin of [
    'https://mlsysbook.ai.attacker.example',
    'https://mlsysbook.ai@attacker.example',
    'https://attacker.example@mlsysbook.ai',
    'https://harvard-edge.github.io.attacker.example',
    'http://localhost:30001',
    'https://mlsysbook.ai:8443',
    'http://mlsysbook.ai',
    'https://mlsysbook.ai/not-an-origin',
    'https://mlsysbook.ai?not=an-origin',
    'https://mlsysbook.ai#not-an-origin',
    'null',
    'not a URL',
  ]) {
    const response = await worker.fetch(new Request('https://analytics.test/', {
      method: 'OPTIONS', headers: { Origin: origin },
    }), environment());
    assert.equal(response.status, 403, origin);
    assert.equal(response.headers.get('Access-Control-Allow-Origin'), null);
  }
});

test('anonymous event ingestion stays available on existing root and routed paths', async () => {
  const env = environment({ ADMIN_SECRET: undefined });
  for (const path of ['/', '/events', '/api/staffml-analytics', '/api/staffml-analytics/events']) {
    const response = await worker.fetch(new Request(`https://analytics.test${path}`, {
      method: 'POST',
      headers: { Origin: 'https://mlsysbook.ai', 'Content-Type': 'application/json' },
      body: JSON.stringify({ events: [{ type: 'page_view', page: '/staffml/' }] }),
    }), env);
    assert.equal(response.status, 200, path);
    assert.deepEqual(await response.json(), { accepted: 1 });
  }
});

test('malformed JSON and bodies without an events array return a client error', async () => {
  for (const body of ['{', 'null', 'false', '42', '"text"', '[]', '{}', '{"events":null}', '{"events":{}}']) {
    const response = await worker.fetch(new Request('https://analytics.test/events', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body,
    }), environment());
    assert.equal(response.status, 400, body);
    assert.ok((await response.json()).error);
  }
});

test('invalid event entries are skipped without dropping valid events in the batch', async () => {
  const response = await worker.fetch(new Request('https://analytics.test/events', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ events: [null, false, 42, 'text', [], {}, { type: 'unknown' }, { type: 'page_view', page: '/staffml/' }] }),
  }), environment());
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), { accepted: 1 });
});

test('nested field values cannot crash ingestion or drop a valid sibling event', async () => {
  const response = await worker.fetch(new Request('https://analytics.test/events', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ events: [
      { type: 'page_view', page: { toString: null } },
      { type: 'page_view', unexpected: { toString: {} } },
      { type: 'question_scored', level: 'L3', score: { toString: null } },
      { type: 'page_view', page: '/valid' },
    ] }),
  }), environment());
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), { accepted: 1 });
});

test('submitted prototype key names remain isolated score buckets', async () => {
  const env = environment();
  const prototypeBefore = Object.getOwnPropertyDescriptors(Object.prototype);
  const submitted = await worker.fetch(new Request('https://analytics.test/events', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ events: ['__proto__', 'constructor', 'toString', 'L3'].map(level => ({
      type: 'question_scored', level, score: 2,
    })) }),
  }), env);
  assert.equal(submitted.status, 200);
  const response = await worker.fetch(new Request('https://analytics.test/summary', {
    headers: { Authorization: 'Bearer test-operator-secret' },
  }), env);
  assert.equal(response.status, 200);
  const { last7Days } = await response.json();
  for (const level of ['__proto__', 'constructor', 'toString', 'L3']) {
    assert.ok(Object.hasOwn(last7Days.scoresByLevel, level));
    assert.deepEqual(last7Days.scoresByLevel[level], { total: 2, count: 1, avg: '2.00' });
  }
  assert.deepEqual(Object.getOwnPropertyDescriptors(Object.prototype), prototypeBefore);
});
