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
// Chip R4-M-3: if the manifest hasn't succeeded in > 24h, don't blindly
// serve from cache forever — bypass cache on question/search endpoints so
// users see current data even if stale-cache was valid.
let lastManifestSuccessMs = 0;
const STALE_MANIFEST_BYPASS_MS = 24 * 60 * 60 * 1000;

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
      // Re-validate against the allowlist in case a compromised earlier
      // session poisoned the IDB entry (Chip R4-C-2 defense-in-depth).
      const persisted = await idbGet("vault_api_origin");
      if (persisted && _isAllowedOrigin(persisted)) {
        self.__VAULT_API_ORIGIN = persisted;
      }
      // Gemini R5-C-2: persist + restore currentRelease so offline SW wake-ups
      // find the correct cache. Previously cacheName was '-unknown' until
      // /manifest succeeded; activate then deleted the real cache on mismatch.
      const persistedRelease = await idbGet("vault_current_release");
      if (persistedRelease && typeof persistedRelease === "string") {
        currentRelease = persistedRelease;
        cacheName = `${CACHE_PREFIX}${VERSION}-${persistedRelease}`;
      }
      // DO NOT prune caches in activate — pruning happens inside
      // updateReleaseFromManifest() only AFTER we've confirmed the new
      // release_id online. Offline activate must never destroy valid cache.
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
    lastManifestSuccessMs = Date.now();
    if (manifest.release_id && manifest.release_id !== currentRelease) {
      currentRelease = manifest.release_id;
      cacheName = `${CACHE_PREFIX}${VERSION}-${manifest.release_id}`;
      // Gemini R5-C-2: persist the release so offline wake-ups can restore it.
      idbSet("vault_current_release", manifest.release_id);
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
      const manifestStale = lastManifestSuccessMs > 0
        && Date.now() - lastManifestSuccessMs > STALE_MANIFEST_BYPASS_MS;
      const cache = await caches.open(cacheName);

      // Chip R7-H-1: hoist `cached` above the manifestStale gate so the
      // offline-fallback path at line "if (cached) return cached" actually
      // has access to the variable. Previously const-scoped inside the if,
      // which made offline fetch-failure crash with ReferenceError — exactly
      // when offline-first was supposed to work.
      let cached = null;
      if (!manifestStale) {
        cached = await cache.match(event.request);
        if (cached) {
          const ageHeader = cached.headers.get("x-sw-cached-at");
          const age = ageHeader ? Date.now() - Number.parseInt(ageHeader, 10) : Infinity;
          if (age < MAX_AGE_MS) return cached;
          // expired; fall through to revalidate (but keep `cached` as offline fallback)
        }
      } else {
        // R4-M-3 bypass path: still grab cached as a last-resort offline
        // fallback; we just won't return it on the fresh-fetch happy path.
        cached = await cache.match(event.request);
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
//
// Hardened per Chip R4-C-2: without these checks, any page on the same
// scope (including a compromised third-party script) could repoint the SW
// to exfiltrate every vault request. Every one of these checks must pass.

// Build-time allowlist of acceptable API origins. These match CORS_ALLOWLIST
// in wrangler.toml and the production/staging worker hostnames.
const VAULT_API_ALLOWLIST = [
  "https://staffml-vault.mlsysbook.ai",
  "https://staffml-vault-staging.mlsysbook.ai",
  // Local dev shim only valid when SW origin is localhost; enforced below.
  "http://localhost:8002",
  "http://127.0.0.1:8002",
];

function _isAllowedOrigin(origin) {
  if (!origin || typeof origin !== "string") return false;
  if (!VAULT_API_ALLOWLIST.includes(origin)) return false;
  // localhost API origins are only permissible when the SW itself lives on
  // localhost (dev environment). Block them on production hosts.
  if (origin.startsWith("http://localhost") || origin.startsWith("http://127.0.0.1")) {
    const swOrigin = self.location.origin;
    if (!swOrigin.startsWith("http://localhost") && !swOrigin.startsWith("http://127.0.0.1")) {
      return false;
    }
  }
  return true;
}

self.addEventListener("message", (event) => {
  // 1. Source must be a same-origin window client, not another worker or
  //    cross-origin document.
  if (!event.source || !("url" in event.source)) return;
  let sourceOrigin;
  try {
    sourceOrigin = new URL(event.source.url).origin;
  } catch {
    return;
  }
  if (sourceOrigin !== self.location.origin) return;

  // 2. Message must be our specific protocol shape.
  if (!event.data || event.data.type !== "SET_VAULT_API_ORIGIN") return;
  const origin = event.data.origin;

  // 3. Origin must be in the build-time allowlist.
  if (!_isAllowedOrigin(origin)) return;

  // 4. Refuse to overwrite once set to a different value without an explicit
  //    reset — prevents a late-loading compromised script from repointing a
  //    session that's already bound.
  if (self.__VAULT_API_ORIGIN && self.__VAULT_API_ORIGIN !== origin) return;

  self.__VAULT_API_ORIGIN = origin;
  idbSet("vault_api_origin", origin);
});
