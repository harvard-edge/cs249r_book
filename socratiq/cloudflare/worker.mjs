/** A bounded Groq chat gateway. No user content, credentials, or provider bodies are logged. */
export const LIMITS = Object.freeze({ bodyBytes: 131072, messageCharacters: 64000,
  totalCharacters: 96000, messages: 32, outputBytes: 262144, maxTokens: 2048,
  bodyTimeoutMs: 5000, requestTimeoutMs: 40000 });
const DEFAULT_LIMITS = LIMITS;
export const GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions';
const MODELS = new Set(['openai/gpt-oss-120b', 'openai/gpt-oss-20b']);
const UNSUPPORTED = new Set(['gemini', 'cerebras', 'sambanova', 'mistral', 'openai', 'huggingface', 'awan']);
// Exact paths only: never strip an arbitrary prefix or decode/rewrite lookalikes.
const CHAT_ROUTES = new Map([
  ['/ai', false], ['/ai/stream', true],
  ['/api/socratiq/ai', false], ['/api/socratiq/ai/stream', true],
]);
const HEALTH_ROUTES = new Set(['/health', '/api/socratiq/health']);
class GatewayError extends Error {
  constructor(status, code) { super(code); this.status = status; }
}
function fail(status, code) { throw new GatewayError(status, code); }
function deadline(milliseconds, signal) {
  const controller = new AbortController();
  let reject;
  const promise = new Promise((_, no) => { reject = no; });
  // A deadline may fire while downstream backpressure prevents a read.
  promise.catch(() => {});
  const stop = code => { reject(new GatewayError(code === 'request_cancelled' ? 499 : 504, code)); controller.abort(); };
  const timer = setTimeout(() => stop('request_timeout'), milliseconds);
  const abort = () => stop('request_cancelled');
  signal?.addEventListener('abort', abort, { once: true });
  if (signal?.aborted) abort();
  return { signal: controller.signal, race: work => Promise.race([work, promise]),
    close() { clearTimeout(timer); signal?.removeEventListener('abort', abort); },
    abort() { controller.abort(); } };
}
async function readBounded(body, maximum, gate) {
  if (!body) fail(400, 'missing_body');
  const reader = body.getReader();
  let size = 0;
  const parts = [];
  try {
    while (true) {
      const { value, done } = await gate.race(reader.read());
      if (done) break;
      size += value.byteLength;
      if (size > maximum) fail(413, 'body_too_large');
      parts.push(value);
    }
  } catch (error) { void reader.cancel().catch(() => {}); throw error; }
  finally { reader.releaseLock(); }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const part of parts) { bytes.set(part, offset); offset += part.byteLength; }
  return bytes;
}
async function readJsonBody(request, limits = DEFAULT_LIMITS, gate) {
  const bytes = await readBounded(request.body, limits.bodyBytes, gate);
  try {
    return JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes));
  } catch {
    fail(400, 'invalid_json');
  }
}
function validateChatPayload(body, stream, limits = DEFAULT_LIMITS) {
  if (!body || typeof body !== 'object' || Array.isArray(body)) fail(400, 'invalid_payload');
  const fields = new Set(['model', 'messages', 'max_tokens', 'stream', 'temperature', 'top_p']);
  if (Object.keys(body).some(key => !fields.has(key))) fail(400, 'unsupported_parameter');
  if (!MODELS.has(body.model)) fail(400, 'unsupported_model');
  if (body.stream !== undefined && typeof body.stream !== 'boolean') fail(400, 'invalid_stream');
  if (body.stream === true && !stream) fail(400, 'stream_requires_stream_route');
  if (!Array.isArray(body.messages) || !body.messages.length || body.messages.length > limits.messages) fail(400, 'invalid_messages');
  let total = 0;
  const messages = body.messages.map(message => {
    if (!message || typeof message !== 'object' || Array.isArray(message)
      || Object.keys(message).some(key => key !== 'role' && key !== 'content')
      || !['system', 'user', 'assistant'].includes(message.role)
      || typeof message.content !== 'string' || !message.content.trim()
      || message.content.length > limits.messageCharacters) fail(400, 'invalid_message');
    total += message.content.length;
    // Preserve instruction text and message order exactly after JSON decoding.
    return { role: message.role, content: message.content };
  });
  if (total > limits.totalCharacters || !messages.some(message => message.role === 'user')) fail(400, 'invalid_messages');
  const maxTokens = body.max_tokens ?? limits.maxTokens;
  if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > limits.maxTokens) fail(400, 'invalid_max_tokens');
  const payload = { model: body.model, messages, max_tokens: maxTokens, stream: stream && body.stream !== false };
  for (const [key, maximum] of [['temperature', 2], ['top_p', 1]]) {
    if (body[key] !== undefined) {
      if (!Number.isFinite(body[key]) || body[key] < 0 || body[key] > maximum) fail(400, `invalid_${key}`);
      payload[key] = body[key];
    }
  }
  return payload;
}
function payloadFor(body, stream, limits = DEFAULT_LIMITS) {
  return validateChatPayload(body, stream, limits);
}
function corsFor(request, env) {
  const origin = request.headers.get('Origin');
  const origins = String(env.ALLOWED_ORIGINS ?? '').split(',').map(item => item.trim()).filter(Boolean);
  if (!origins.length || origins.some(item => {
    try { const parsed = new URL(item); return parsed.origin !== item || !['http:', 'https:'].includes(parsed.protocol); }
    catch { return true; }
  })) fail(503, 'origins_not_configured');
  if (!origin || !origins.includes(origin)) fail(403, 'origin_not_allowed');
  return { 'Access-Control-Allow-Origin': origin, 'Vary': 'Origin', 'Cache-Control': 'no-store',
    'X-Content-Type-Options': 'nosniff' };
}
function jsonError(status, code, cors = {}, extra = {}) {
  return new Response(JSON.stringify({ error: { code, message: code.replaceAll('_', ' ') } }), {
    status, headers: { ...cors, ...extra, 'Content-Type': 'application/json; charset=utf-8' },
  });
}
function boundedStream(body, gate, headers, limits = DEFAULT_LIMITS) {
  const reader = body.getReader();
  let size = 0;
  let finished = false;
  const finish = () => { finished = true; gate.close(); reader.releaseLock(); };
  return new Response(new ReadableStream({
    async pull(controller) {
      if (finished) return;
      try {
        const { value, done } = await gate.race(reader.read());
        if (done) { finish(); controller.close(); return; }
        size += value.byteLength;
        if (size > limits.outputBytes) throw new GatewayError(502, 'upstream_output_too_large');
        controller.enqueue(value);
      } catch (error) {
        gate.abort();
        void reader.cancel().catch(() => {});
        finish();
        // Headers are already sent. An SSE error lets both clients reject partial output.
        const code = error instanceof GatewayError ? error.message : 'upstream_stream_failed';
        controller.enqueue(new TextEncoder().encode(`\ndata: ${JSON.stringify({ error: { code } })}\n\n`));
        controller.close();
      }
    },
    async cancel() {
      if (finished) return;
      gate.abort();
      await reader.cancel().catch(() => {});
      finish();
    },
  }), { headers });
}
export function createWorker({ fetchImpl = globalThis.fetch, limits = DEFAULT_LIMITS } = {}) {
  const resolvedLimits = { ...DEFAULT_LIMITS, ...limits };
  return { async fetch(request, env) {
    let cors = {};
    let gate;
    try {
      const url = new URL(request.url);
      // Dispatch/liveness only, including when disabled. No config, secret,
      // limiter, or provider lookup; no configuration details in the response.
      if (HEALTH_ROUTES.has(url.pathname)) {
        const headers = { 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' };
        if (!['GET', 'HEAD'].includes(request.method)) {
          return jsonError(405, 'method_not_allowed', headers, { Allow: 'GET, HEAD' });
        }
        return new Response(null, { status: 204, headers });
      }
      cors = corsFor(request, env);
      if (!CHAT_ROUTES.has(url.pathname)) fail(404, 'route_not_found');
      if (request.method === 'OPTIONS') {
        const method = request.headers.get('Access-Control-Request-Method');
        const requested = (request.headers.get('Access-Control-Request-Headers') ?? '').toLowerCase().split(',').map(s => s.trim()).filter(Boolean);
        if (method !== 'POST' || requested.some(header => !['content-type', 'accept'].includes(header))) fail(403, 'preflight_not_allowed');
        return new Response(null, { status: 204, headers: { ...cors, 'Access-Control-Allow-Methods': 'POST, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type, Accept', 'Access-Control-Max-Age': '600' } });
      }
      if (request.method !== 'POST') return jsonError(405, 'method_not_allowed', cors, { Allow: 'POST, OPTIONS' });
      if (env.ENABLED !== 'true') fail(503, 'gateway_disabled');
      if (url.searchParams.getAll('provider').length !== 1 || url.searchParams.getAll('url').length > 1
        || [...url.searchParams.keys()].some(key => !['provider', 'url'].includes(key))) fail(400, 'invalid_provider_query');
      const provider = url.searchParams.get('provider');
      if (UNSUPPORTED.has(provider)) fail(501, 'provider_not_supported');
      if (provider !== 'groq') fail(400, 'unknown_provider');
      if (url.searchParams.has('url') && url.searchParams.get('url') !== GROQ_URL) fail(400, 'target_not_allowed');
      if (typeof env.GROQ_API_KEY !== 'string' || !env.GROQ_API_KEY.trim() || !env.AI_RATE_LIMITER?.limit) fail(503, 'gateway_not_configured');
      if (!(request.headers.get('Content-Type') ?? '').toLowerCase().match(/^application\/json(?:\s*;|$)/)) fail(415, 'json_required');
      const length = request.headers.get('Content-Length');
      if (length && (!/^\d+$/.test(length) || Number(length) > resolvedLimits.bodyBytes)) fail(413, 'body_too_large');
      // CF-Connecting-IP is supplied by Cloudflare; missing IPs share a conservative bucket.
      gate = deadline(resolvedLimits.requestTimeoutMs, request.signal);
      const rate = await gate.race(env.AI_RATE_LIMITER.limit({ key: request.headers.get('CF-Connecting-IP') ?? 'missing-ip' }));
      if (!rate?.success) {
        gate.close();
        return jsonError(429, 'rate_limit_exceeded', cors, { 'Retry-After': '60' });
      }
      const bodyGate = deadline(resolvedLimits.bodyTimeoutMs, gate.signal);
      let body;
      try {
        body = await gate.race(readJsonBody(request, resolvedLimits, bodyGate));
      } finally { bodyGate.close(); }
      const payload = payloadFor(body, CHAT_ROUTES.get(url.pathname), resolvedLimits);
      const upstream = await gate.race(fetchImpl(GROQ_URL, {
        method: 'POST', redirect: 'manual', signal: gate.signal,
        headers: { Authorization: `Bearer ${env.GROQ_API_KEY}`, 'Content-Type': 'application/json',
          Accept: payload.stream ? 'text/event-stream' : 'application/json' },
        body: JSON.stringify(payload),
      }));
      if (!upstream.ok) {
        void upstream.body?.cancel().catch(() => {});
        gate.close();
        const status = upstream.status === 429 ? 429 : upstream.status >= 500 ? 502
          : upstream.status >= 400 ? upstream.status : 502;
        // Do not expose provider response bodies or authentication diagnostics.
        const retry = upstream.headers.get('Retry-After');
        return jsonError(status, upstream.status === 429 ? 'upstream_rate_limited' : 'upstream_rejected', cors,
          retry && /^\d{1,5}$/.test(retry) ? { 'Retry-After': retry } : {});
      }
      if (!upstream.body) fail(502, 'upstream_missing_body');
      const type = upstream.headers.get('Content-Type')?.split(';')[0].trim().toLowerCase();
      if (payload.stream) {
        if (type !== 'text/event-stream') fail(502, 'upstream_content_type');
        return boundedStream(upstream.body, gate, { ...cors, 'Content-Type': 'text/event-stream; charset=utf-8' }, resolvedLimits);
      }
      if (type !== 'application/json') fail(502, 'upstream_content_type');
      let bytes;
      try { bytes = await readBounded(upstream.body, resolvedLimits.outputBytes, gate); }
      catch (error) { if (error.message === 'body_too_large') fail(502, 'upstream_output_too_large'); throw error; }
      let result;
      try { result = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes)); }
      catch { fail(502, 'upstream_invalid_json'); }
      if (result.error || typeof result.choices?.[0]?.message?.content !== 'string'
        || !result.choices[0].message.content.trim()) fail(502, 'upstream_invalid_response');
      gate.close();
      return new Response(bytes, { headers: { ...cors, 'Content-Type': 'application/json; charset=utf-8' } });
    } catch (error) {
      gate?.abort(); gate?.close();
      return jsonError(error instanceof GatewayError ? error.status : 502,
        error instanceof GatewayError ? error.message : 'gateway_unavailable', cors);
    }
  } };
}
export default createWorker();
