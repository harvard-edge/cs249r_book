import { SOCRATIQ_API_BASE, SHARED_PROVIDER, MAX_OUTPUT_TOKENS } from '../../src_shared/provider-config.js';
export { MAX_OUTPUT_TOKENS };

// ─── Backend Toggle ──────────────────────────────────────────────────────────
// Set to true to use local Cloudflare Workers (wrangler dev), false for prod
export const USE_LOCAL_WORKERS = false;

// ─── Worker URLs ─────────────────────────────────────────────────────────────
const URL_SERVER_QUIZ_DEV = "http://localhost:8787"
const URL_SERVER_QUIZ_DEV_STREAMING = "http://localhost:8788"
const URL_SERVER_QUIZ_PROD = SOCRATIQ_API_BASE
const URL_SERVER_QUIZ_STREAMING = SOCRATIQ_API_BASE

export const WORKER_URL_AI = USE_LOCAL_WORKERS ? URL_SERVER_QUIZ_DEV + "/ai" : URL_SERVER_QUIZ_PROD + "/ai"
export const WORKER_URL_AI_STREAM = USE_LOCAL_WORKERS ? URL_SERVER_QUIZ_DEV_STREAMING + "/ai/stream" : URL_SERVER_QUIZ_STREAMING + "/ai/stream"
export const BASEURL = "https://tinymlbackend3.azurewebsites.net/"
export const SERVELESSURLJSON = "https://tinymljson.azurewebsites.net/api/GroqJsonFunction";

// Updated to use our new Cloudflare proxy
export const SERVELESSURLGROQ = URL_SERVER_QUIZ_PROD + "/ai?url=https://api.groq.com/openai/v1/chat/completions&provider=groq"
export const SERVERLESSGEMINI = URL_SERVER_QUIZ_PROD + "/ai?url=https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent&provider=gemini"
export const SERVERLESSRRESULTS = "https://proxy-worker.mlsysbook.workers.dev/results" // Existing results service, outside the AI gateway
export const SERVERLESSSCORE = "https://proxy-worker.mlsysbook.workers.dev/score"

// Updated DuckAI to use our new streaming proxy
export const SERVELESSURLDuckAI = URL_SERVER_QUIZ_STREAMING + "/ai/stream?url=https://api.groq.com/openai/v1/chat/completions&provider=groq";
export const SIZE_LIMIT_LLM_CALL = 6000; // Adjust this value as needed
export const STORAGE_KEY_CHAPTERS = 'chapter_progress_data';
export const APP_SECRET = 'MyS1SML_v!_' + new Date().getFullYear();

// ─── Site Identity ────────────────────────────────────────────────────────────
// Change this to match the textbook/site this widget is embedded in
export const MAIN_TOPIC = 'MLSysBook.AI: Principles and Practices of Machine Learning Systems Engineering';

// ─── AI Provider Models ───────────────────────────────────────────────────────
// Controls which model each provider uses — edit here to swap models globally
// The supported shared gateway uses Groq; other providers remain explicit calls.
export const DEFAULT_PROVIDER = SHARED_PROVIDER.name.toUpperCase();
export const PROVIDER_MODELS = {
  GROQ:        { model: SHARED_PROVIDER.studyModel,                        stream: SHARED_PROVIDER.stream },
  GEMINI:      { model: 'gemini-2.5-flash',                                  stream: true  },
  CEREBRAS:    { model: 'gpt-oss-120b',                                      stream: true  },
  SAMBANOVA:   { model: 'gpt-oss-120b',                                      stream: true  },
  MISTRAL:     { model: 'mistral-tiny',                                      stream: true  },
  OPENAI:      { model: 'deepseek/deepseek-chat-v3.1:free',                  stream: true  },
  HUGGINGFACE: { model: 'meta-llama/Llama-3.1-8B-Instruct:featherless-ai',  stream: false },
  AWAN:        { model: 'Meta-Llama-3.1-70B-Instruct',                       stream: true  },
};
