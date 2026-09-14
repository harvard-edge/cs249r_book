/**
 * PWA manifest regression tests.
 *
 * Guards the contract between src/app/manifest.ts and the raster icons in
 * public/icons/: every icon the manifest declares must exist on disk with
 * exactly the pixel dimensions it advertises. A silent mismatch (e.g. an
 * icon regenerated at the wrong size, or a renamed file) doesn't fail the
 * build — it just breaks installability in the field, which is precisely
 * the kind of drift a test should catch.
 */
import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import path from "node:path";
import manifest from "@/app/manifest";

const PUBLIC_DIR = path.resolve(__dirname, "../../../public");

/** Width/height from a PNG's IHDR chunk (bytes 16–23, big-endian). */
function pngDimensions(file: string): { width: number; height: number } {
  const buf = readFileSync(file);
  // PNG signature: 89 50 4E 47 0D 0A 1A 0A
  expect(buf.subarray(0, 8).toString("hex")).toBe("89504e470d0a1a0a");
  return { width: buf.readUInt32BE(16), height: buf.readUInt32BE(20) };
}

describe("PWA manifest", () => {
  const m = manifest();

  it("has the fields installability requires", () => {
    expect(m.name).toBeTruthy();
    expect(m.short_name).toBe("StaffML");
    expect(m.display).toBe("standalone");
    expect(m.start_url).toBe("/");
    // scope must contain start_url or the browser rejects the pair.
    expect(m.start_url!.startsWith(m.scope!)).toBe(true);
  });

  it("declares both any and maskable icons at 192 and 512", () => {
    for (const purpose of ["any", "maskable"] as const) {
      const sizes = (m.icons ?? [])
        .filter((i) => i.purpose === purpose)
        .map((i) => i.sizes);
      expect(sizes, `purpose=${purpose}`).toEqual(
        expect.arrayContaining(["192x192", "512x512"]),
      );
    }
  });

  it.each((manifest().icons ?? []).map((i) => [i.src, i.sizes] as const))(
    "icon %s exists in public/ at its declared size %s",
    (src, sizes) => {
      const file = path.join(PUBLIC_DIR, src.replace(/^\//, ""));
      const [w, h] = sizes!.split("x").map(Number);
      expect(pngDimensions(file)).toEqual({ width: w, height: h });
    },
  );

  it("ships the apple-touch-icon referenced by layout metadata", () => {
    const file = path.join(PUBLIC_DIR, "icons", "apple-touch-icon.png");
    expect(pngDimensions(file)).toEqual({ width: 180, height: 180 });
  });
});
