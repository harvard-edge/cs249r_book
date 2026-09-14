import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const serviceWorker = readFileSync(
  path.resolve(__dirname, "../../../public/sw.js"),
  "utf8",
);

describe("PWA service worker cache lifetime", () => {
  it("finishes a navigation cache write before returning the response", () => {
    expect(serviceWorker).toContain(
      "if (res && res.ok) await cache.put(event.request, res.clone());",
    );
  });

  it("keeps cached-asset revalidation alive through cache maintenance", () => {
    expect(serviceWorker).toContain("event.waitUntil(network);");
    expect(serviceWorker).toContain("await cache.put(request, res.clone());");
    expect(serviceWorker).toContain(
      "await appTrimCache(APP_ASSET_CACHE, APP_ASSET_CACHE_LIMIT);",
    );
  });
});
