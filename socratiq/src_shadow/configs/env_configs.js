const URL_SERVER_QUIZ_DEV = "http://localhost:8787"
const URL_SERVER_QUIZ_PROD = "https://proxy-worker.mlsysbook.workers.dev"
const URL_SERVER_QUIZ_STREAMING = "https://proxy-worker-streaming.mlsysbook.workers.dev"
export const BASEURL = "https://tinymlbackend3.azurewebsites.net/"
export const SERVELESSURLJSON = "https://tinymljson.azurewebsites.net/api/GroqJsonFunction";

// Updated to use our new Cloudflare proxy
export const SERVELESSURLGROQ = URL_SERVER_QUIZ_PROD + "/ai?url=https://api.groq.com/openai/v1/chat/completions&provider=groq"
export const SERVERLESSGEMINI = URL_SERVER_QUIZ_PROD + "/ai?url=https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent&provider=gemini"
export const SERVERLESSRRESULTS = URL_SERVER_QUIZ_PROD + "/results" // Keep existing results endpoint
export const SERVERLESSSCORE = URL_SERVER_QUIZ_PROD + "/score" // Keep existing score endpoint

// Updated DuckAI to use our new streaming proxy
export const SERVELESSURLDuckAI = URL_SERVER_QUIZ_STREAMING + "/ai/stream?url=https://api.groq.com/openai/v1/chat/completions&provider=groq";
export const SIZE_LIMIT_LLM_CALL = 6000; // Adjust this value as needed
export const STORAGE_KEY_CHAPTERS = 'chapter_progress_data';
export const APP_SECRET = 'MyS1SML_v!_' + new Date().getFullYear();