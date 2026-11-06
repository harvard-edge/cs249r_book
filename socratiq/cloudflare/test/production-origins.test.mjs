// Configured-origin policy with the actual gateway and constructed upstreams.
// These checks make no provider requests and do not measure Cloudflare capacity.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createWorker, GROQ_URL} from '../worker.mjs';

const config = readFileSync(new URL('../wrangler.toml', import.meta.url), 'utf8');
const varsSection = config.split(/^\[vars\]\s*$/m)[1]?.split(/^\[/m)[0];
assert.ok(varsSection, 'The production configuration must declare its origin policy.');
// These policy variables deliberately use simple quoted TOML strings. A format
// change requires updating this reader along with the deployment-policy review.
const variable = name => {
  const match = varsSection.match(new RegExp(`^${name}\\s*=\\s*("(?:\\\\.|[^"\\\\])*")\\s*$`, 'm'));
  assert.ok(match, `${name} must have one explicit quoted policy value.`);
  return JSON.parse(match[1]);
};
const productionVars = {ENABLED: variable('ENABLED'), ALLOWED_ORIGINS: variable('ALLOWED_ORIGINS')};
const origins = ['https://mlsysbook.ai', 'https://harvard-edge.github.io'];
const endpoint = 'https://mlsysbook.ai/api/socratiq/ai';
const payload = {model: 'openai/gpt-oss-120b', messages: [{role: 'user', content: 'Constructed origin-policy check.'}], max_tokens: 1, stream: false};
// Positive transport probes enable only their constructed environment. A real
// ENABLED=false configuration must still pass policy checks and be deployable.
const env = limiter => ({...productionVars, ENABLED: 'true', GROQ_API_KEY: 'constructed-origin-test-key', AI_RATE_LIMITER: limiter ?? {limit: async () => ({success: true})}});
function request(origin, {method = 'POST', stream = false, headers = {}} = {}) {
  return new Request(`${endpoint}${stream ? '/stream' : ''}?provider=groq`, {
    method,
    headers: {
      ...(origin === undefined ? {} : {Origin: origin}),
      ...(method === 'POST' ? {'Content-Type': 'application/json'} : {}),
      ...headers,
    },
    ...(method === 'POST' ? {body: JSON.stringify({...payload, stream})} : {}),
  });
}
function assertCors(response, origin) {
  assert.equal(response.headers.get('Access-Control-Allow-Origin'), origin);
  assert.equal(response.headers.get('Vary'), 'Origin');
  assert.equal(response.headers.get('Access-Control-Allow-Credentials'), null);
  assert.equal(response.headers.get('Cache-Control'), 'no-store');
}

test('production config grants only the exact production and development origins', () => {
  assert.deepEqual(productionVars.ALLOWED_ORIGINS.split(','), origins);
  assert.ok(['true', 'false'].includes(productionVars.ENABLED));
});

for (const origin of origins) {
  test(`${origin}: preflight echoes exact origin without touching key, quota, or provider`, async () => {
    const config = new Proxy({...productionVars}, {
      get(target, key) {
        assert.equal(key, 'ALLOWED_ORIGINS', 'Preflight must not inspect credentials, enable state, or quota.');
        return target[key];
      },
    });
    const worker = createWorker({fetchImpl: async () => assert.fail('Preflight cannot contact a provider.')});
    const response = await worker.fetch(request(origin, {method: 'OPTIONS', headers: {
      'Access-Control-Request-Method': 'POST', 'Access-Control-Request-Headers': 'content-type,accept',
    }}), config);
    assert.equal(response.status, 204);
    assertCors(response, origin);
    assert.equal(response.headers.get('Access-Control-Allow-Methods'), 'POST, OPTIONS');
  });

  test(`${origin}: JSON and streaming POST retain fixed provider policy and exact CORS`, async () => {
    for (const stream of [false, true]) {
      let calls = 0;
      const worker = createWorker({fetchImpl: async (url, options) => {
        calls++;
        assert.equal(url, GROQ_URL);
        assert.equal(options.redirect, 'manual');
        assert.equal(options.headers.Authorization, 'Bearer constructed-origin-test-key');
        const body = JSON.parse(options.body);
        assert.deepEqual(body.messages, payload.messages);
        assert.equal(body.stream, stream);
        return stream
          ? new Response('data: {"choices":[{"delta":{"content":"constructed answer"}}]}\n\ndata: [DONE]\n\n', {headers: {'Content-Type': 'text/event-stream'}})
          : Response.json({choices: [{message: {content: 'constructed answer'}}]});
      }});
      const response = await worker.fetch(request(origin, {stream}), env());
      assert.equal(response.status, 200);
      assertCors(response, origin);
      assert.match(await response.text(), /constructed answer/);
      assert.equal(calls, 1);
    }
  });
}

const rejectedOrigins = [
  undefined, 'null', 'https://attacker.example',
  'https://mlsysbook.ai.attacker.example', 'https://harvard-edge.github.io.attacker.example',
  'https://other-owner.github.io', 'http://harvard-edge.github.io',
  'https://harvard-edge.github.io:8443', 'https://harvard-edge.github.io/cs249r_book_dev/',
];
for (const origin of rejectedOrigins) {
  test(`unapproved origin is rejected before quota/provider: ${String(origin)}`, async () => {
    const worker = createWorker({fetchImpl: async () => assert.fail('Denied origins cannot reach the provider.')});
    const config = new Proxy({...productionVars}, {
      get(target, key) {
        assert.equal(key, 'ALLOWED_ORIGINS', 'Denied origins must not inspect other bindings.');
        return target[key];
      },
    });
    for (const method of ['OPTIONS', 'POST']) {
      const response = await worker.fetch(request(origin, {method}), config);
      assert.equal(response.status, 403);
      assert.equal((await response.json()).error.code, 'origin_not_allowed');
      assert.equal(response.headers.get('Access-Control-Allow-Origin'), null);
    }
  });
}

test('both origins consume the same connecting-IP quota key', async () => {
  const keys = [];
  let calls = 0;
  const limiter = {limit: async ({key}) => {keys.push(key); return {success: keys.length <= 2};}};
  const worker = createWorker({fetchImpl: async () => {calls++; return Response.json({choices: [{message: {content: 'constructed answer'}}]});}});
  const config = env(limiter);
  for (const [index, origin] of [origins[0], origins[1], origins[0], origins[1]].entries()) {
    const response = await worker.fetch(request(origin, {headers: {'CF-Connecting-IP': '192.0.2.7'}}), config);
    assert.equal(response.status, index < 2 ? 200 : 429);
    assertCors(response, origin);
    if (index >= 2) assert.equal(response.headers.get('Retry-After'), '60');
    await response.text();
  }
  assert.deepEqual(keys, Array(4).fill('192.0.2.7'));
  assert.equal(calls, 2);
});

test('both origins retain the disable switch and missing-limiter fail-closed checks', async () => {
  const worker = createWorker({fetchImpl: async () => assert.fail('Unavailable gateway cannot contact a provider.')});
  for (const origin of origins) {
    for (const [config, error] of [[{...env(), ENABLED: 'false'}, 'gateway_disabled'], [{...env(), AI_RATE_LIMITER: undefined}, 'gateway_not_configured']]) {
      const response = await worker.fetch(request(origin), config);
      assert.equal(response.status, 503);
      assert.equal((await response.json()).error.code, error);
      assertCors(response, origin);
    }
  }
});

test('GitHub project paths are not an authentication boundary and Referer grants nothing', async () => {
  const worker = createWorker({fetchImpl: async () => Response.json({choices: [{message: {content: 'constructed answer'}}]})});
  for (const page of ['/cs249r_book_dev/vol1/', '/unrelated-repository/']) {
    const response = await worker.fetch(request(origins[1], {headers: {Referer: `${origins[1]}${page}`}}), env());
    assert.equal(response.status, 200);
    await response.text();
  }
  const denied = await worker.fetch(request(undefined, {headers: {Referer: `${origins[1]}/cs249r_book_dev/vol1/`}}), env());
  assert.equal(denied.status, 403);
  assert.equal((await denied.json()).error.code, 'origin_not_allowed');
});
