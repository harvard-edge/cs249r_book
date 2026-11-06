// Constructed transport responses: no external requests or model quality claims.
import test from 'node:test';
import assert from 'node:assert/strict';
import {callCloudflareWithLegacyFormat, callProviderStream, tryMultipleProvidersStream, tryMultipleProvidersSingle} from '../src_shadow/js/libs/agents/cloudflareAgent.js';
import {CustomAPIAdapter} from '../src_shadow/js/libs/agents/custom-api-adapter.js';
const collect = async generator => {let result = ''; for await (const chunk of generator) result += chunk; return result;};
const frame = text => `data: ${JSON.stringify({choices:[{delta:{content:text}}]})}\n`;
const originalFetch = globalThis.fetch;
const originalStorage = globalThis.localStorage;
const originalWindow = globalThis.window;
test.afterEach(() => {globalThis.fetch = originalFetch; globalThis.localStorage = originalStorage; globalThis.window = originalWindow;});
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
test('custom API streaming failure never forwards the prompt to the hosted service', async () => {
  settings({customAPI:{enabled:true,endpoint:'https://constructed.invalid',provider:'openai'}});
  const called = [];
  globalThis.fetch = async url => {called.push(url); return new URL(url).hostname === 'constructed.invalid' ? new Response('Constructed failure', {status:503}) : new Response(frame('Constructed fallback') + 'data: [DONE]\n');};
  await assert.rejects(collect(tryMultipleProvidersStream({prompt:'Constructed private prompt'})), /custom API request failed/);
  assert.equal(called.length, 1);
  assert.equal(new URL(called[0]).hostname, 'constructed.invalid');
});

test('custom API nonstream failure never forwards the prompt to the hosted service', async () => {
  settings({customAPI:{enabled:true,endpoint:'http://127.0.0.1:11434/api/generate',provider:'ollama'}});
  const called = [];
  globalThis.fetch = async url => {called.push(url); throw new Error('Constructed private connection failure');};
  await assert.rejects(tryMultipleProvidersSingle({prompt:'Constructed private prompt'}), /disable the custom API/);
  assert.deepEqual(called, ['http://127.0.0.1:11434/api/generate']);
});

test('DuckAI custom-service answer and flashcard failures do not use a hosted fallback', async () => {
  settings({customAPI:{enabled:true,endpoint:'http://127.0.0.1:11434/api/generate',provider:'ollama'}});
  const {DuckAI} = await import('../src_shadow/js/libs/agents/duck-ai-cloudflare.js');
  for (const method of ['generateAnswer','generateFlashcards']) {
    const called = [];
    globalThis.fetch = async url => {called.push(url); return new Response('Constructed failure',{status:503});};
    await assert.rejects(collect(new DuckAI()[method]('Constructed private source')), /custom API request failed/);
    assert.deepEqual(called, ['http://127.0.0.1:11434/api/generate']);
  }
});

test('enabled incomplete custom settings fail before any request', async () => {
  settings({customAPI:{enabled:true,endpoint:'',provider:'openai'}});
  let requested = false;
  globalThis.fetch = async () => {requested=true;};
  await assert.rejects(collect(tryMultipleProvidersStream({prompt:'Constructed private prompt'})), /Complete your custom API settings/);
  assert.equal(requested, false);
});

test('legacy quiz wrappers never bypass a failed selected custom service', async () => {
  settings({customAPI:{enabled:true,endpoint:'http://127.0.0.1:11434/api/generate',provider:'ollama'}});
  for (const streaming of [false,true]) {
    const called = [];
    globalThis.fetch = async url => {called.push(url); throw new Error('Constructed private failure');};
    const params = {prompt:'Constructed private quiz task',quote:'Constructed private source',context:{title:'Constructed',url:'https://constructed.invalid',position:{scrollY:0}}};
    await assert.rejects(async () => {
      const result = await callCloudflareWithLegacyFormat(params, 'groq', streaming);
      if (streaming) await collect(result);
    }, /custom API request failed/);
    assert.deepEqual(called, ['http://127.0.0.1:11434/api/generate']);
  }
});

test('legacy quiz wrapper parses the selected custom service response', async () => {
  settings({customAPI:{enabled:true,endpoint:'https://constructed.invalid/chat',provider:'openai'}});
  const called = [];
  globalThis.fetch = async url => {called.push(url); return Response.json({choices:[{message:{content:JSON.stringify({questions:[{question:'Constructed custom question',answers:[{text:'Constructed',correct:true}]}]})},finish_reason:'stop'}]});};
  const result = await callCloudflareWithLegacyFormat({prompt:'Constructed quiz task',quote:'Constructed source',context:{title:'Constructed',url:'https://constructed.invalid',position:{scrollY:0}}});
  assert.equal(result[0].question, 'Constructed custom question');
  assert.deepEqual(called, ['https://constructed.invalid/chat']);
});

test('separate system policy reaches the configured default without removal', async () => {
  settings({}); let request;
  globalThis.fetch = async (url, options) => { request = JSON.parse(options.body); return new Response(frame('Constructed answer') + 'data: [DONE]\n'); };
  await collect(tryMultipleProvidersStream({prompt:'Constructed user prompt',systemPrompt:'Constructed system policy'}));
  assert.ok(request.messages[0].content.startsWith('Constructed system policy\n\n'));
  assert.match(request.messages[0].content, /Learner calibration/);
  assert.deepEqual(request.messages.at(-1), {role:'user',content:'Constructed user prompt'});
  assert.equal(request.max_tokens, 2048);
});

test('default legacy streaming uses the public gateway and quoted shared learner preferences', async () => {
  const records = {userSettings: JSON.stringify({sliderValue: 0}), socratiq_learner_preferences_v1: JSON.stringify({version:1,level:'advanced',background:'Constructed background',instructions:'Constructed preference'})};
  globalThis.localStorage = {getItem: key => records[key] ?? null};
  let request;
  globalThis.fetch = async (url, options) => {request = {url:new URL(url),body:JSON.parse(options.body)}; return new Response(frame('Constructed answer') + 'data: [DONE]\n');};
  await collect(tryMultipleProvidersStream({prompt:'Constructed task'}));
  assert.equal(request.url.origin + request.url.pathname, 'https://mlsysbook.ai/api/socratiq/ai/stream');
  assert.match(request.body.messages[0].content, /comfortable with machine learning and systems concepts/);
  assert.ok(!request.body.messages[0].content.includes('Constructed preference'));
  assert.match(request.body.messages[1].content, /Constructed preference/);
});

test('explicit custom endpoint keeps policy, conversation, and learner calibration', async () => {
  settings({customAPI:{enabled:true,endpoint:'https://constructed.invalid/chat',provider:'openai',model:'constructed-model'}});
  let request;
  globalThis.fetch = async (url, options) => {request = {url,body:JSON.parse(options.body)}; return new Response(frame('Constructed answer') + 'data: [DONE]\n');};
  await collect(tryMultipleProvidersStream({prompt:'Constructed task',systemPrompt:'Constructed policy',messages:[{role:'user',content:'Prior task'},{role:'assistant',content:'Prior answer'},{role:'user',content:'Constructed task'}],learnerPreferences:{level:'beginner',instructions:'Constructed request preference'}}));
  assert.equal(request.url, 'https://constructed.invalid/chat');
  assert.equal(request.body.model, 'constructed-model');
  assert.ok(request.body.messages[0].content.startsWith('Constructed policy\n\n'));
  assert.match(request.body.messages[0].content, /new to machine learning systems/);
  assert.deepEqual(request.body.messages.slice(-3), [{role:'user',content:'Prior task'},{role:'assistant',content:'Prior answer'},{role:'user',content:'Constructed task'}]);
  assert.match(request.body.messages[1].content, /Constructed request preference/);
});

test('existing custom provider formats preserve separate system policy and conversation', () => {
  const messages = [{role:'system',content:'Constructed policy'},{role:'user',content:'Constructed task'},{role:'assistant',content:'Constructed attempt'},{role:'user',content:'Constructed follow-up'}];
  const config = provider => ({provider,model:'constructed-model'});
  const gemini = CustomAPIAdapter._formatRequestForProvider(messages, config('google-gemini'));
  assert.deepEqual(gemini.systemInstruction, {parts:[{text:'Constructed policy'}]});
  assert.deepEqual(gemini.contents.map(item => item.role), ['user','model','user']);
  const anthropic = CustomAPIAdapter._formatRequestForProvider(messages, config('anthropic'));
  assert.equal(anthropic.system, 'Constructed policy');
  assert.deepEqual(anthropic.messages, messages.slice(1));
  const ollama = CustomAPIAdapter._formatRequestForProvider(messages, config('ollama'));
  assert.equal(ollama.system, 'Constructed policy');
  assert.match(ollama.prompt, /Constructed task[\s\S]*Constructed attempt[\s\S]*Constructed follow-up/);
});

test('session-only custom key reaches Authorization without needing a persisted key', async () => {
  const persisted = {customAPI:{enabled:true,endpoint:'https://constructed.invalid/chat',provider:'openai',model:'constructed-model',apiKey:'',saveLocally:false}};
  const saved = JSON.stringify(persisted);
  globalThis.localStorage = {getItem: key => key === 'userSettings' ? saved : null};
  globalThis.window = {customAPISettings:{...persisted.customAPI,apiKey:'constructed-session-key'}};
  let authorization;
  globalThis.fetch = async (_url, options) => {authorization = options.headers.Authorization; return new Response(frame('Constructed answer') + 'data: [DONE]\n');};
  await collect(tryMultipleProvidersStream({prompt:'Constructed task'}));
  assert.equal(authorization, 'Bearer constructed-session-key');
  assert.equal(JSON.parse(saved).customAPI.apiKey, '');
  const {DuckAI} = await import('../src_shadow/js/libs/agents/duck-ai-cloudflare.js');
  assert.equal(new DuckAI().getCurrentSettings().customAPI.apiKey, 'constructed-session-key');
});

test('stale opted-out stored keys are ignored and session disable wins over storage', () => {
  settings({customAPI:{enabled:true,endpoint:'https://constructed.invalid',provider:'openai',apiKey:'constructed-stale-key',saveLocally:false}});
  globalThis.window = {};
  assert.equal(CustomAPIAdapter.getRuntimeSettings().customAPI.apiKey, '');
  globalThis.window.customAPISettings = {enabled:false,endpoint:'https://constructed.invalid',provider:'openai',apiKey:'constructed-session-key'};
  assert.equal(CustomAPIAdapter.hasCustomAPIConfig(CustomAPIAdapter.getRuntimeSettings()), false);
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
