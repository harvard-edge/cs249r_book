// Constructed transport responses: no external requests or model quality claims.
import test from 'node:test';
import assert from 'node:assert/strict';
import {callCloudflareWithLegacyFormat, callProviderStream, tryMultipleProvidersStream} from '../src_shadow/js/libs/agents/cloudflareAgent.js';
const collect = async generator => {let result = ''; for await (const chunk of generator) result += chunk; return result;};
const frame = text => `data: ${JSON.stringify({choices:[{delta:{content:text}}]})}\n`;
const originalFetch = globalThis.fetch;
const originalStorage = globalThis.localStorage;
test.afterEach(() => {globalThis.fetch = originalFetch; globalThis.localStorage = originalStorage;});
const settings = config => {globalThis.localStorage = {getItem: () => JSON.stringify(config)};};
test('legacy Gemini quizzes route to Gemini with its configured model', async () => {
  let request;
  globalThis.fetch = async (url, options) => {
    request = {url: new URL(url), body: JSON.parse(options.body)};
    return Response.json({candidates:[{content:{parts:[{text:JSON.stringify({questions:[{question:'Constructed question',answers:[{text:'A',correct:true}]}]})}]}}]});
  };
  const result = await callCloudflareWithLegacyFormat({prompt:'Constructed quiz request',quote:'Constructed source passage',context:{title:'Constructed context',url:'https://constructed.invalid',position:{scrollY:0}}}, 'gemini');
  assert.equal(request.url.searchParams.get('provider'), 'gemini');
  assert.match(request.url.searchParams.get('url'), /generativelanguage.googleapis.com/);
  assert.equal(request.body.model, 'gemini-2.5-flash');
  assert.equal(request.body.max_tokens, 2048);
  assert.equal(result[0].question, 'Constructed question');
});
test('stream reads split SSE and final data without a trailing newline', async () => {
  globalThis.fetch = async () => new Response(new ReadableStream({start(controller) {
    const bytes = new TextEncoder().encode(frame('Constructed ') + frame('answer') + 'data: [DONE]');
    controller.enqueue(bytes.slice(0, 15)); controller.enqueue(bytes.slice(15)); controller.close();
  }}), {headers:{'content-type':'text/event-stream'}});
  assert.equal(await collect(callProviderStream('Constructed prompt', 'groq')), 'Constructed answer');
});
test('an empty completed default stream fails without trying unsupported providers', async () => {
  settings({}); const called = [];
  globalThis.fetch = async url => {called.push(new URL(url).searchParams.get('provider')); return new Response('data: [DONE]\n');};
  await assert.rejects(collect(tryMultipleProvidersStream({prompt:'Constructed prompt'})), /No content/);
  assert.deepEqual(called, ['groq']);
});
test('a failed partial stream is surfaced without appending a second provider', async () => {
  settings({}); const called = [];
  globalThis.fetch = async url => {called.push(url); return new Response(frame('Constructed partial') + 'data: {"error":{"message":"Constructed failure"}}\n');};
  let observed = '';
  await assert.rejects(async () => {for await (const chunk of tryMultipleProvidersStream({prompt:'Constructed prompt'})) observed += chunk;}, /Constructed failure/);
  assert.equal(observed, 'Constructed partial'); assert.equal(called.length, 1);
});
test('disabled custom API does not suppress the configured provider', async () => {
  settings({customAPI:{enabled:false,endpoint:'https://constructed.invalid',provider:'openai'}});
  let requested;
  globalThis.fetch = async url => {requested = url; return new Response(frame('Constructed answer') + 'data: [DONE]\n');};
  assert.equal(await collect(tryMultipleProvidersStream({prompt:'Constructed prompt'})), 'Constructed answer');
  assert.notEqual(new URL(requested).hostname, 'constructed.invalid');
});
test('custom API failure falls back before any content', async () => {
  settings({customAPI:{enabled:true,endpoint:'https://constructed.invalid',provider:'openai'}});
  const called = [];
  globalThis.fetch = async url => {called.push(url); return new URL(url).hostname === 'constructed.invalid' ? new Response('Constructed failure', {status:503}) : new Response(frame('Constructed fallback') + 'data: [DONE]\n');};
  assert.equal(await collect(tryMultipleProvidersStream({prompt:'Constructed prompt'})), 'Constructed fallback');
  assert.equal(called.length, 2);
});

test('separate system policy reaches the configured default without removal', async () => {
  settings({}); let request;
  globalThis.fetch = async (url, options) => { request = JSON.parse(options.body); return new Response(frame('Constructed answer') + 'data: [DONE]\n'); };
  await collect(tryMultipleProvidersStream({prompt:'Constructed user prompt',systemPrompt:'Constructed system policy'}));
  assert.deepEqual(request.messages, [{role:'system',content:'Constructed system policy'},{role:'user',content:'Constructed user prompt'}]);
  assert.equal(request.max_tokens, 2048);
});

test('completed flashcards collect all split JSON tokens before returning cards', async () => {
  const {DuckAI} = await import('../src_shadow/js/libs/agents/duck-ai-cloudflare.js');
  const cards = [{question:'Constructed question',answer:'Constructed answer'}];
  const content = JSON.stringify(cards);
  let request;
  globalThis.fetch = async (url, options) => {
    request = JSON.parse(options.body);
    return new Response(frame(content.slice(0, 17)) + frame(content.slice(17)) + 'data: [DONE]\n');
  };
  const generator = new DuckAI()._generateFlashcardsStreaming('Constructed prompt', {name:'groq'});
  let chunks = '', result;
  while (true) {
    const next = await generator.next();
    if (next.done) {result = next.value; break;}
    chunks += next.value;
  }
  assert.equal(chunks, content);
  assert.deepEqual(result, cards);
  assert.equal(request.model, 'openai/gpt-oss-20b');
  assert.equal(request.max_tokens, 2048);
});
