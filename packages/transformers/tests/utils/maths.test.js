import { medianFilter, bankers_round, log_softmax } from "../../src/utils/maths.js";
import { init } from "../init.js";

// Initialise the testing environment
init();

describe("Mathematical operations", () => {
  describe("bankers rounding", () => {
    it("should round up to nearest even", () => {
      expect(bankers_round(-0.5)).toBeCloseTo(0);
      expect(bankers_round(1.5)).toBeCloseTo(2);
      expect(bankers_round(19.5)).toBeCloseTo(20);
    });
    it("should round down to nearest even", () => {
      expect(bankers_round(-1.5)).toBeCloseTo(-2);
      expect(bankers_round(2.5)).toBeCloseTo(2);
      expect(bankers_round(18.5)).toBeCloseTo(18);
    });
  });

  describe("median filtering", () => {
    it("should compute median filter", async () => {
      const t1 = new Float32Array([5, 12, 2, 6, 3, 10, 9, 1, 4, 8, 11, 7]);
      const window = 3;

      const target = new Float32Array([12, 5, 6, 3, 6, 9, 9, 4, 4, 8, 8, 11]);

      const output = medianFilter(t1, window);
      expect(output).toBeCloseToNested(target, 5);
    });

    // TODO add tests for errors
  });

  describe("log softmax", () => {
    // Should match output of scipy log_softmax
    it("should compute log softmax correctly for usual values", () => {
      const input = [0, 1, 2, 3];
      const expected = [-3.4401896985611953, -2.4401896985611953, -1.4401896985611953, -0.44018969856119533];
      const output = log_softmax(input);
      expect(output).toBeCloseToNested(expected, 10);
    });

    it("should compute log softmax correctly for values with large differences", () => {
      const input = [1000, 1];
      const expected = [0, -999];
      const output = log_softmax(input);
      expect(output).toBeCloseToNested(expected, 10);
    });
  });
});
