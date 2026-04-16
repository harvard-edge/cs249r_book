/**
 * Service worker — release-keyed cache for vault API responses.
 *
 * Closes REVIEWS.md convergent finding (Chip N-H2 / Dean N-6 / David N-4):
 * SW cache entries include release_id in the key so a deploy atomically
 * evicts stale content. TTL 7d so offline users don't see month-stale
 * questions.
 *
 * This worker registers passively — registration wired by the site's layout
 * only after the Phase-4 cutover. Before cutover, registration is a no-op
 * because `vault-api.ts` isn't the data source yet.
 */

const VERSION = "v1";
const CACHE_PREFIX = "staffml-vault-";
const MAX_AGE_MS = 7 * 24 * 60 * 60 * 1000;   // 7 days

let currentRelease = null;   // learned from /manifest
let cacheName = `${CACHE_PREFIX}${VERSION}-unknown`;

self.addEventListener("install", (event) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      await self.clients.claim();
      // Evict any caches that are not the current one; release-change invalidation.
      const names = await caches.keys();
      await Promise.all(
        names
          .filter(n => n.startsWith(CACHE_PREFIX) && n !== cacheName)
          .map(n => caches.delete(n)),
      );
    })(),
  );
});

async function updateReleaseFromManifest(api) {
  try {
    const res = await fetch(`${api}/manifest`, { cache: "no-store" });
    if (!res.ok) return;
    const manifest = await res.json();
    if (manifest.release_id && manifest.release_id !== currentRelease) {
      currentRelease = manifest.release_id;
      cacheName = `${CACHE_PREFIX}${VERSION}-${manifest.release_id}`;
      const names = await caches.keys();
      await Promise.all(
        names
          .filter(n => n.startsWith(CACHE_PREFIX) && n !== cacheName)
          .map(n => caches.delete(n)),
      );
    }
  } catch {
    // manifest failure is fine — SW keeps serving whatever it has.
  }
}

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  const api = self.__VAULT_API_ORIGIN;
  if (!api) return;
  if (!url.href.startsWith(api)) return;
  if (event.request.method !== "GET") return;

  event.respondWith(
    (async () => {
      await updateReleaseFromManifest(api);
      const cache = await caches.open(cacheName);

      const cached = await cache.match(event.request);
      if (cached) {
        const ageHeader = cached.headers.get("x-sw-cached-at");
        const age = ageHeader ? Date.now() - Number.parseInt(ageHeader, 10) : Infinity;
        if (age < MAX_AGE_MS) return cached;
        // expired; fall through to revalidate
      }

      try {
        const res = await fetch(event.request);
        if (res.ok) {
          const body = await res.clone().blob();
          const headers = new Headers(res.headers);
          headers.set("x-sw-cached-at", String(Date.now()));
          await cache.put(event.request, new Response(body, {
            status: res.status,
            statusText: res.statusText,
            headers,
          }));
        }
        return res;
      } catch (err) {
        if (cached) return cached; // offline fallback
        throw err;
      }
    })(),
  );
});

// Site posts the API origin on registration; worker won't intercept fetches until it's set.
self.addEventListener("message", (event) => {
  if (event.data && event.data.type === "SET_VAULT_API_ORIGIN") {
    self.__VAULT_API_ORIGIN = event.data.origin;
  }
});
