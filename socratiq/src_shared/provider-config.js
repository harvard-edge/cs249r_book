// Public routing and model names only. Provider credentials belong to the gateway.
export const SOCRATIQ_API_BASE = 'https://mlsysbook.ai/api/socratiq';
export const MAX_OUTPUT_TOKENS = 2048;

// The public gateway currently supports this provider. Adding a provider requires
// a server adapter and verification before it can become a browser default.
export const SHARED_PROVIDER = Object.freeze({
  name: 'groq',
  target: 'https://api.groq.com/openai/v1/chat/completions',
  readerModel: 'openai/gpt-oss-120b',
  studyModel: 'openai/gpt-oss-20b',
  stream: true,
});
