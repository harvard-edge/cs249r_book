// Author-constructed browser fixture. No model calls or measured learning claims.
import {createServer} from 'node:http';
import {readFile} from 'node:fs/promises';

const bundle = new URL('../../books/shared/assets/scripts/socratiq/reader.js', import.meta.url);
const contextModule = new URL('../src_reader/context.js', import.meta.url);
createServer(async (request, response) => {
  const pathname = new URL(request.url, 'http://localhost').pathname;
  if (pathname === '/reader.js' || pathname === '/context.js') {
    try { response.setHeader('Content-Type', 'text/javascript'); response.end(await readFile(pathname === '/reader.js' ? bundle : contextModule)); }
    catch { response.writeHead(404); response.end('Build the reader first.'); }
    return;
  }
  if (!/^\/vol[1-4]\/chapter\/$/.test(pathname)) {response.writeHead(404); response.end(); return;}
  const volume = pathname.split('/')[1];
  response.setHeader('Content-Type', 'text/html');
  response.end(`<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Reader test chapter</title>
  <style>body{font:18px/1.7 Georgia,serif;color:#252525;background:#fff;margin:0}main{max-width:720px;margin:64px auto;padding:24px}nav{padding:12px;background:#eee}h1{line-height:1.2}p{margin:24px 0} @media(prefers-color-scheme:dark){body{color:#eee;background:#1b1b1b}nav{background:#333}}</style>
  <script src="/reader.js" data-volume="${volume}" defer></script></head><body><nav>Navigation must not be sent to the model.</nav>
  <main id="quarto-document-content"><h1>Understanding memory bandwidth</h1><section id="bandwidth"><h2>Bandwidth and latency</h2>
  <p id="passage">Memory bandwidth measures how much data can move per unit time. Latency measures how long an individual request takes.</p>
  <p>A system can have high bandwidth and still have high latency. A wider road carries more cars without shortening the journey.</p>
  <p id="math">Transfer time is <span class="katex"><span>duplicated visual math</span><math><semantics><annotation encoding="application/x-tex">t = D / B</annotation></semantics></math></span>, where D is data volume and B is bandwidth.</p>
  </section><section id="other"><h2>Another section</h2><p>Unrelated content should not be included in the first section's context.</p></section></main></body></html>`);
}).listen(4176, '127.0.0.1');
