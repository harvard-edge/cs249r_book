import { describe, expect, test } from "vitest";
import { QUESTION_COUNT_DISPLAY, roundedFloorForDisplay } from "../corpus";

describe("QUESTION_COUNT_DISPLAY (Bug 7)", () => {
  test("never renders the '0+' bug and is well-formed", () => {
    expect(QUESTION_COUNT_DISPLAY).not.toBe("0+");
    expect(QUESTION_COUNT_DISPLAY).toMatch(/^\d{1,3}(,\d{3})*\+$/);
  });

  test("rounds down to a sensible granularity by magnitude", () => {
    expect(roundedFloorForDisplay(8053)).toBe(8000);
    expect(roundedFloorForDisplay(9999)).toBe(9000);
    expect(roundedFloorForDisplay(1000)).toBe(1000);
    // Below 1000 the old code produced 0; now it degrades gracefully.
    expect(roundedFloorForDisplay(950)).toBe(900);
    expect(roundedFloorForDisplay(90)).toBe(90);
    expect(roundedFloorForDisplay(7)).toBe(7);
    expect(roundedFloorForDisplay(0)).toBe(0);
  });
});
