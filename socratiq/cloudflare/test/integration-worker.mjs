// Constructed upstream responses only. This fixture never calls a provider.
import { createWorker } from '../worker.mjs';
const worker = createWorker({ fetchImpl: async (_url, options) => {
  const payload = JSON.parse(options.body);
  const content = JSON.stringify(payload.messages);
  if (!payload.stream) return Response.json({ choices: [{ message: { content } }] });
  const data = { choices: [{ delta: { content }, finish_reason: null }] };
  return new Response(`data: ${JSON.stringify(data)}\n\ndata: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n`,
    { headers: { 'Content-Type': 'text/event-stream' } });
} });
export default { fetch(request, env) { return worker.fetch(request, { ...env, GROQ_API_KEY: 'constructed-local-test-key' }); } };
