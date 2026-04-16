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

let currentRelease = null;          // learned from /manifest
let cacheName = `${CACHE_PREFIX}${VERSION}-unknown`;

// Manifest-polling throttle (Chip R3-C2): don't fetch /manifest on every
// intercepted request — that doubles worker load and nullifies the §10.4
// cost model. Cap at once per 5 minutes; release-change latency is well
// under the §6.1.1 10-minute grace window.
const MANIFEST_POLL_MS = 5 * 60 * 1000;
let lastManifestCheckMs = 0;

// Persist the API origin in IDB so SW restarts don't lose it (Chip R3-H3).
// Otherwise a cold tab opened offline hits a no-op fetch handler.
const IDB_NAME = "staffml-sw";
const IDB_STORE = "config";

function openIdb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(IDB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(IDB_STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function idbGet(key) {
  try {
    const db = await openIdb();
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_STORE, "readonly").objectStore(IDB_STORE).get(key);
      tx.onsuccess = () => resolve(tx.result);
      tx.onerror = () => reject(tx.error);
    });
  } catch {
    return null;
  }
}

async function idbSet(key, value) {
  try {
    const db = await openIdb();
    await new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_STORE, "readwrite").objectStore(IDB_STORE).put(value, key);
      tx.onsuccess = resolve;
      tx.onerror = () => reject(tx.error);
    });
  } catch {
    /* swallow */
  }
}

self.addEventListener("install", (event) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      await self.clients.claim();
      // Rehydrate the API origin from IDB so a cold wake-up can serve cached
      // responses before any page posts SET_VAULT_API_ORIGIN (Chip R3-H3).
      const persisted = await idbGet("vault_api_origin");
      if (persisted) self.__VAULT_API_ORIGIN = persisted;
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
  // TTL-throttle to once per 5 minutes (Chip R3-C2). Previous per-request
  // polling doubled worker load and nullified the §10.4 cost model.
  const now = Date.now();
  if (now - lastManifestCheckMs < MANIFEST_POLL_MS) return;
  lastManifestCheckMs = now;
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

// Site posts the API origin on registration; SW persists it so a later
// cold-wake-up (especially offline) survives without the page's help.
self.addEventListener("message", (event) => {
  if (event.data && event.data.type === "SET_VAULT_API_ORIGIN") {
    self.__VAULT_API_ORIGIN = event.data.origin;
    idbSet("vault_api_origin", event.data.origin);
  }
});
