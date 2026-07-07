/*
 * TinyTorch service worker — offline support for the PWA.
 *
 * Strategy:
 *   - Navigations (HTML): network-first. The site deploys frequently, so
 *     pages must never go stale; the cache is only a fallback for offline
 *     revisits, with a branded offline page as the last resort.
 *   - Same-origin static assets (css/js/images/fonts): stale-while-revalidate.
 *     One deploy of staleness is acceptable for assets and keeps repeat
 *     visits fast.
 *   - Cross-origin requests (Google Fonts, cdnjs) are left to the browser —
 *     caching opaque responses inflates storage quota unpredictably.
 *
 * Bump CACHE_VERSION when changing caching behavior; old caches are dropped
 * on activate.
 */

'use strict';

const CACHE_VERSION = 'v1';
const PAGE_CACHE = `tinytorch-pages-${CACHE_VERSION}`;
const ASSET_CACHE = `tinytorch-assets-${CACHE_VERSION}`;
const ASSET_CACHE_LIMIT = 200;

// Everything the offline fallback needs is inlined below, so install only
// warms the cache with the manifest and install-prompt icons.
const PRECACHE_URLS = [
  './manifest.webmanifest',
  './assets/images/icons/icon-192.png',
  './assets/images/icons/icon-512.png',
];

const OFFLINE_HTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Offline — TinyTorch</title>
<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    display: flex; align-items: center; justify-content: center;
    min-height: 100vh; margin: 0; background: #fff; color: #333;
    text-align: center; padding: 1rem;
  }
  @media (prefers-color-scheme: dark) {
    body { background: #1a1a1a; color: #ddd; }
  }
  .flame { font-size: 4rem; }
  h1 { margin: 0.5rem 0; font-size: 1.5rem; }
  p { color: #888; max-width: 28rem; }
  button {
    margin-top: 1rem; padding: 0.5rem 1.5rem; font-size: 1rem;
    background: #D4740C; color: #fff; border: none; border-radius: 6px;
    cursor: pointer;
  }
</style>
</head>
<body>
<main>
  <div class="flame">🔥</div>
  <h1>You're offline</h1>
  <p>This TinyTorch page isn't cached yet. Pages you've visited before are
  available offline — reconnect to load this one.</p>
  <button onclick="location.reload()">Try again</button>
</main>
</body>
</html>`;

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(ASSET_CACHE)
      .then((cache) => cache.addAll(PRECACHE_URLS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      const names = await caches.keys();
      await Promise.all(
        names
          .filter((n) => n.startsWith('tinytorch-') && n !== PAGE_CACHE && n !== ASSET_CACHE)
          .map((n) => caches.delete(n))
      );
      if (self.registration.navigationPreload) {
        await self.registration.navigationPreload.enable();
      }
      await self.clients.claim();
    })()
  );
});

async function trimCache(cacheName, limit) {
  const cache = await caches.open(cacheName);
  const keys = await cache.keys();
  if (keys.length <= limit) return;
  // Cache keys are ordered oldest-first; drop from the front.
  await Promise.all(keys.slice(0, keys.length - limit).map((k) => cache.delete(k)));
}

async function networkFirstPage(event) {
  const cache = await caches.open(PAGE_CACHE);
  try {
    const response = (await event.preloadResponse) || (await fetch(event.request));
    if (response && response.ok) {
      cache.put(event.request, response.clone());
    }
    return response;
  } catch (err) {
    const cached = await cache.match(event.request);
    if (cached) return cached;
    return new Response(OFFLINE_HTML, {
      status: 503,
      headers: { 'Content-Type': 'text/html; charset=utf-8' },
    });
  }
}

async function staleWhileRevalidate(request) {
  const cache = await caches.open(ASSET_CACHE);
  const cached = await cache.match(request);
  const network = fetch(request)
    .then((response) => {
      if (response && response.ok) {
        cache.put(request, response.clone());
        trimCache(ASSET_CACHE, ASSET_CACHE_LIMIT);
      }
      return response;
    })
    .catch(() => undefined);
  return cached || network.then((r) => r || Response.error());
}

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET') return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  if (request.mode === 'navigate') {
    event.respondWith(networkFirstPage(event));
    return;
  }

  const dest = request.destination;
  if (dest === 'style' || dest === 'script' || dest === 'image' || dest === 'font') {
    event.respondWith(staleWhileRevalidate(request));
  }
  // Everything else (search.json, release-manifest.json, XHR/fetch data)
  // goes straight to the network so dynamic data is never served stale.
});
