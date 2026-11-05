import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { createExplanationRequest } from '../../src_reader/transport.mjs';

const origin = 'https://book.example';
const port = Number(process.env.WORKER_TEST_PORT ?? 8799);
const base = `http://127.0.0.1:${port}`;
const config = fileURLToPath(new URL('./wrangler.toml', import.meta.url));
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const child = spawn('wrangler', ['dev', '--local', '--ip', '127.0.0.1', '--port', String(port), '--config', config, '--show-interactive-dev-session=false'], {
  stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env, WRANGLER_SEND_METRICS: 'false' },
});
let output = '';
child.stdout.on('data', chunk => { output += chunk.toString(); });
child.stderr.on('data', chunk => { output += chunk.toString(); });
async function call(path, body, headers = {}) {
  return fetch(`${base}${path}`, { method: 'POST', headers: { Origin: origin, 'Content-Type': 'application/json', ...headers }, body: JSON.stringify(body) });
}

test('local Workers runtime exercises reader, legacy JSON, origin rejection, and rate binding with constructed upstream', { timeout: 45000 }, async () => {
  try {
    let ready = false;
    for (let attempt = 0; attempt < 100; attempt++) {
      if (child.exitCode !== null) throw new Error(`Wrangler exited: ${output}`);
      try { await fetch(`${base}/ai`, { headers: { Origin: origin } }); ready = true; break; } catch {}
      await sleep(150);
    }
    assert.ok(ready, `Local worker did not become ready: ${output}`);
    const explanation = createExplanationRequest({ endpoint: `${base}/ai/stream`, fetchImpl: (url, options) => fetch(url, {
      ...options, headers: { ...options.headers, Origin: origin },
    }) });
    const answer = await explanation({ source: { quote: 'SOURCE_SENTINEL Ω', title: 'Constructed fixture', volume: 'I' }, question: 'QUESTION_SENTINEL' });
    const messages = JSON.parse(answer);
    assert.equal(messages[0].role, 'system');
    assert.match(messages[0].content, /You are SocratiQ/);
    assert.match(messages[1].content, /SOURCE_SENTINEL Ω/);
    assert.match(messages[1].content, /QUESTION_SENTINEL/);
    const body = { model: 'openai/gpt-oss-20b', messages: [{ role: 'system', content: '  LOCAL_SYSTEM\nΩ ' }, { role: 'user', content: 'LOCAL_USER' }], stream: false };
    const response = await call('/ai?provider=groq', body);
    assert.equal(response.status, 200);
    assert.deepEqual(JSON.parse((await response.json()).choices[0].message.content), body.messages);
    const denied = await call('/ai?provider=groq', body, { Origin: 'https://attacker.example' });
    assert.equal(denied.status, 403);
    assert.equal(denied.headers.get('Access-Control-Allow-Origin'), null);
    const statuses = [];
    for (let attempt = 0; attempt < 15; attempt++) statuses.push((await call('/ai?provider=groq', body)).status);
    assert.ok(statuses.includes(429), `Local rate binding did not reject: ${JSON.stringify(statuses)}`);
    assert.ok(statuses.every(status => status === 200 || status === 429));
    console.log(JSON.stringify({ upstream: 'constructed stub; no provider egress', statuses }));
  } finally {
    child.kill('SIGTERM');
    if (child.exitCode === null) await Promise.race([new Promise(resolve => child.once('exit', resolve)), sleep(3000)]);
    console.log(output);
  }
});
