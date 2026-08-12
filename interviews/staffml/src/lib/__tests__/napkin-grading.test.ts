import { describe, expect, test } from "vitest";
import {
  checkNapkinMath,
  extractFinalNumber,
  extractFinalQuantity,
  gradeNapkinAnswer,
} from "../corpus";

describe("checkNapkinMath — sub-unit answers (Bug 3)", () => {
  test("a 2x miss on a sub-unit answer is NOT graded 'within tolerance'", () => {
    // model 0.5, user 1.0 → true relative error 100%. The old code divided by
    // max(0.5,1)=1, scoring ratio 0.5 → "ballpark". With |model| it is 1.0.
    const r = checkNapkinMath(1.0, 0.5, "cloud");
    expect(r.ratio).toBeCloseTo(1.0, 5);
    expect(r.grade).toBe("ballpark"); // ratio <= 1.0 boundary, but no longer "close"
    expect(["exact", "close"]).not.toContain(r.grade);
  });

  test("exact match on a sub-unit answer grades 'exact'", () => {
    const r = checkNapkinMath(0.5, 0.5, "cloud");
    expect(r.ratio).toBe(0);
    expect(r.grade).toBe("exact");
  });

  test("within-tolerance on a sub-unit answer still grades 'close' or better", () => {
    // cloud tolerance 0.25; 0.55 vs 0.5 → 10% error → within tolerance.
    const r = checkNapkinMath(0.55, 0.5, "cloud");
    expect(r.ratio).toBeCloseTo(0.1, 5);
    expect(["exact", "close"]).toContain(r.grade);
  });

  test("large-magnitude answers keep working", () => {
    const r = checkNapkinMath(1_050_000, 1_000_000, "cloud");
    expect(r.ratio).toBeCloseTo(0.05, 5);
    expect(["exact", "close"]).toContain(r.grade);
  });

  test("model answer of 0: nonzero guess is way off with a finite label", () => {
    const r = checkNapkinMath(5, 0, "cloud");
    expect(r.grade).toBe("way_off");
    expect(r.label).toBe("Way off"); // not "Off by Infinity×"
  });

  test("model answer of 0 and user 0 is exact", () => {
    const r = checkNapkinMath(0, 0, "cloud");
    expect(r.grade).toBe("exact");
  });
});

describe("extractFinalNumber — garbage rejection (Bug 4)", () => {
  test("a lone comma does NOT parse to 0", () => {
    expect(extractFinalNumber(",")).toBeNull();
    expect(extractFinalNumber("the cache, the model, the data")).toBeNull();
  });

  test("returns the last numeric token", () => {
    expect(extractFinalNumber("first 12 then 34")).toBe(34);
  });

  test("strips thousands separators", () => {
    expect(extractFinalNumber("about 1,024 GB")).toBe(1024);
  });

  test("honors an explicit answer marker", () => {
    expect(extractFinalNumber("scratch work 99\nanswer: 42")).toBe(42);
    expect(extractFinalNumber("=> 3.5")).toBe(3.5);
  });

  test("a marker with no digit falls through instead of yielding 0", () => {
    expect(extractFinalNumber("answer: ,")).toBeNull();
  });

  test("a hyphen range reads as the trailing positive number, not a negative", () => {
    expect(extractFinalNumber("10-20 ms")).toBe(20);
  });

  test("decimals survive", () => {
    expect(extractFinalNumber("the result is 0.5 GB")).toBe(0.5);
  });
});

describe("extractFinalQuantity — unit-aware extraction (A2)", () => {
  test("parses the trailing unit-bearing quantity", () => {
    const q = extractFinalQuantity("Time = 140 / 3350 ≈ 42 ms");
    expect(q).not.toBeNull();
    expect(q!.qty.toBase().scalar).toBeCloseTo(0.042, 6);
    expect(q!.unit).toBe("ms");
  });

  test("skips non-unit trailing tokens — '1.31 MB across 40 layers' → 1.31 MB", () => {
    const q = extractFinalQuantity("Per-layer footprint is 1.31 MB across 40 layers");
    expect(q).not.toBeNull();
    expect(q!.unit).toBe("MB");
    // 1.31 MB in bytes (decimal MB), not 40.
    expect(q!.qty.toBase().scalar).toBeCloseTo(1.31e6, 0);
  });

  test("honors an explicit answer marker line", () => {
    const q = extractFinalQuantity("scratch 99 GB/s\nanswer: 42 ms");
    expect(q!.unit).toBe("ms");
  });

  test("returns null when no token parses to a real unit", () => {
    expect(extractFinalQuantity("about 40 layers and 12 epochs")).toBeNull();
  });
});

describe("gradeNapkinAnswer — unit-aware grading (A2)", () => {
  test("'42 ms' vs '42000 µs' grades as equivalent (exact/close)", () => {
    const r = gradeNapkinAnswer("=> 42 ms", "= 42000 µs", "cloud");
    expect(r).not.toBeNull();
    expect(r!.unitAware).toBe(true);
    expect(["exact", "close"]).toContain(r!.grade);
    expect(r!.userDisplay).toBe("42 ms");
    expect(r!.modelDisplay).toBe("42000 µs");
  });

  test("'100 GB/s' vs '100 GB' is wrong-quantity (incompatible dims)", () => {
    const r = gradeNapkinAnswer("100 GB/s", "100 GB", "cloud");
    expect(r).not.toBeNull();
    expect(r!.grade).toBe("way_off");
    expect(r!.label).toMatch(/Wrong quantity/);
    expect(r!.maxSelfScore).toBe(1);
  });

  test("a milliwatt answer vs a megawatt model must NOT grade equal", () => {
    // mW and MW are dimensionally compatible (power) but 9 orders of magnitude
    // apart, so the SI-prefix distinction must be honored: this can never grade
    // as equivalent. (checkNapkinMath's relative error is taken against the
    // model magnitude; a tiny user vs a huge model saturates near ratio 1, so
    // the grade is "ballpark" rather than "way_off" — the point is it is not
    // exact/close.) We assert both directions to cover the asymmetry.
    const r = gradeNapkinAnswer("=> 5 mW", "=> 5 MW", "cloud");
    expect(r).not.toBeNull();
    expect(r!.unitAware).toBe(true);
    expect(["exact", "close"]).not.toContain(r!.grade);

    // Reverse: a megawatt answer to a milliwatt model is grossly over → way_off.
    const rev = gradeNapkinAnswer("=> 5 MW", "=> 5 mW", "cloud");
    expect(rev!.grade).toBe("way_off");
  });

  test("'1.31 MB across 40 layers' parses 1.31 MB, not 40, for grading", () => {
    const r = gradeNapkinAnswer(
      "Footprint is 1.31 MB across 40 layers",
      "About 1.3 MB",
      "cloud",
    );
    expect(r).not.toBeNull();
    expect(r!.unitAware).toBe(true);
    expect(["exact", "close"]).toContain(r!.grade);
  });

  test("falls back to bare-number grading when no units present", () => {
    const r = gradeNapkinAnswer("first 12 then 34", "the answer is 34", "cloud");
    expect(r).not.toBeNull();
    expect(r!.unitAware).toBe(false);
    expect(r!.grade).toBe("exact");
  });

  test("returns null when neither side has a parseable number or quantity", () => {
    expect(gradeNapkinAnswer("the cache", "the model", "cloud")).toBeNull();
  });
});
