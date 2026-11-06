// Local DuckAI -> production worker.fetch -> constructed provider failure.
import test from 'node:test';
import assert from 'node:assert/strict';
import {createWorker,GROQ_URL} from '../cloudflare/worker.mjs';
import {DuckAI} from '../src_shadow/js/libs/agents/duck-ai-cloudflare.js';
const origin='https://constructed-book.invalid';
const provider={name:'groq',url:GROQ_URL,model:'openai/gpt-oss-20b',streaming:true};
const env={ENABLED:'true',ALLOWED_ORIGINS:origin,GROQ_API_KEY:'constructed-key',AI_RATE_LIMITER:{limit:async()=>({success:true})}};
function install(body){const worker=createWorker({fetchImpl:async()=>new Response(body,{headers:{'Content-Type':'text/event-stream'}})});return async(url,options)=>worker.fetch(new Request(url,{...options,headers:{...options.headers,Origin:origin}}),env);}
const frame=(text,reason=null)=>'data: '+JSON.stringify({choices:[{delta:{content:text},finish_reason:reason}]})+'\n\n';
for(const [name,ending]of [['explicit error','data: {"error":{"code":"request_timeout"}}\n\n'],['EOF',''],['length',frame('','length')+'data: [DONE]\n\n']]){
 test(`DuckAI answer rejects ${name} instead of accepting partial output`,async()=>{const previous=globalThis.fetch;globalThis.fetch=install(frame('Constructed partial answer.')+ending);try{await assert.rejects(async()=>{for await(const chunk of new DuckAI()._generateAnswerStreaming('Constructed prompt',provider))void chunk;});}finally{globalThis.fetch=previous;}});
 test(`DuckAI flashcards reject ${name} even when prefix happens to parse as JSON`,async()=>{const previous=globalThis.fetch;globalThis.fetch=install(frame('[{"question":"Constructed question","answer":"Constructed answer"}]')+ending);try{await assert.rejects(async()=>{for await(const chunk of new DuckAI()._generateFlashcardsStreaming('Constructed prompt',provider))void chunk;});}finally{globalThis.fetch=previous;}});
}
