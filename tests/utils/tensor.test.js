import { Tensor, cat, mean, stack, layer_norm } from "../../src/transformers.js";
import { compare } from "../test_utils.js";

describe("Tensor operations", () => {
  describe("cat", () => {
    it("should concatenate on dim=0", async () => {
      const t1 = new Tensor("float32", [1, 2, 3], [1, 3]);
      const t2 = new Tensor("float32", [4, 5, 6, 7, 8, 9], [2, 3]);
      const t3 = new Tensor("float32", [10, 11, 12], [1, 3]);

      const target1 = new Tensor("float32", [1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3]);
      const target2 = new Tensor("float32", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [4, 3]);

      // 2 tensors
      const concatenated1 = cat([t1, t2], 0);
      compare(concatenated1, target1, 1e-3);

      // 3 tensors
      const concatenated2 = cat([t1, t2, t3], 0);
      compare(concatenated2, target2, 1e-3);
    });

    it("should concatenate on dim=1", async () => {
      const t1 = new Tensor("float32", [1, 2, 3, -1, -2, -3], [2, 3, 1]);
      const t2 = new Tensor("float32", [4, -4], [2, 1, 1]);
      const t3 = new Tensor("float32", [5, 6, -5, -6], [2, 2, 1]);

      const target1 = new Tensor("float32", [1, 2, 3, 4, -1, -2, -3, -4], [2, 4, 1]);
      const target2 = new Tensor("float32", [1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6], [2, 6, 1]);

      // 2 tensors
      const concatenated1 = cat([t1, t2], 1);
      compare(concatenated1, target1, 1e-3);

      // 3 tensors
      const concatenated2 = cat([t1, t2, t3], 1);
      compare(concatenated2, target2, 1e-3);
    });

    it("should concatenate on dim=-2", async () => {
      const t1 = new Tensor("float32", [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16], [2, 1, 3, 2]);
      const t2 = new Tensor("float32", [7, 8, 9, 10, 17, 18, 19, 20], [2, 1, 2, 2]);

      const target = new Tensor("float32", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], [2, 1, 5, 2]);

      const concatenated = cat([t1, t2], -2);

      compare(concatenated, target, 1e-3);
    });

    // TODO add tests for errors
  });

  describe("slice", () => {
    it("should return a given row dim", async () => {
      const t1 = new Tensor("float32", [1, 2, 3, 4, 5, 6], [3, 2]);
      const t2 = t1.slice(1);
      const target = new Tensor("float32", [3, 4], [2]);

      compare(t2, target);
    });

    it("should return a range of rows", async () => {
      const t1 = new Tensor("float32", [1, 2, 3, 4, 5, 6], [3, 2]);
      const t2 = t1.slice([1, 3]);
      const target = new Tensor("float32", [3, 4, 5, 6], [2, 2]);

      compare(t2, target);
    });

    it("should return a crop", async () => {
      const t1 = new Tensor("float32", Array.from({ length: 28 }, (_, i) => i + 1), [4, 7]);
      const t2 = t1.slice([1, -1], [1, -1]);

      const target = new Tensor("float32", [9, 10, 11, 12, 13, 16, 17, 18, 19, 20], [2, 5]);

      compare(t2, target);
    });
  });

  describe("stack", () => {
    const t1 = new Tensor("float32", [0, 1, 2, 3, 4, 5], [1, 3, 2]);

    it("should stack on dim=0", async () => {
      const target1 = new Tensor("float32", [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], [2, 1, 3, 2]);
      const target2 = new Tensor("float32", [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], [3, 1, 3, 2]);

      // 2 tensors
      const stacked1 = stack([t1, t1], 0);
      compare(stacked1, target1, 1e-3);

      // 3 tensors
      const stacked2 = stack([t1, t1, t1], 0);
      compare(stacked2, target2, 1e-3);
    });

    it("should stack on dim=1", async () => {
      const target1 = new Tensor("float32", [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], [1, 2, 3, 2]);
      const target2 = new Tensor("float32", [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], [1, 3, 3, 2]);

      // 2 tensors
      const stacked1 = stack([t1, t1], 1);
      compare(stacked1, target1, 1e-3);

      // 3 tensors
      const stacked2 = stack([t1, t1, t1], 1);
      compare(stacked2, target2, 1e-3);
    });

    it("should stack on dim=-1", async () => {
      const target1 = new Tensor("float32", [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], [1, 3, 2, 2]);
      const target2 = new Tensor("float32", [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5], [1, 3, 2, 3]);

      // 2 tensors
      const stacked1 = stack([t1, t1], -1);
      compare(stacked1, target1, 1e-3);

      // 3 tensors
      const stacked2 = stack([t1, t1, t1], -1);
      compare(stacked2, target2, 1e-3);
    });
  });

  describe("permute", () => {
    it("should permute", async () => {
      const x = new Tensor("float32", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], [2, 3, 4]);
      // Permute axes to (0, 1, 2) - No change
      const permuted_1 = x.permute(0, 1, 2);
      const target_1 = x;
      compare(permuted_1, target_1, 1e-3);

      // Permute axes to (0, 2, 1)
      const permuted_2 = x.permute(0, 2, 1);
      const target_2 = new Tensor("float32", [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19, 23], [2, 4, 3]);
      compare(permuted_2, target_2, 1e-3);

      // Permute axes to (1, 0, 2)
      const permuted_3 = x.permute(1, 0, 2);
      const target_3 = new Tensor("float32", [0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 16, 17, 18, 19, 8, 9, 10, 11, 20, 21, 22, 23], [3, 2, 4]);
      compare(permuted_3, target_3, 1e-3);

      // Permute axes to (1, 2, 0)
      const permuted_4 = x.permute(1, 2, 0);
      const target_4 = new Tensor("float32", [0, 12, 1, 13, 2, 14, 3, 15, 4, 16, 5, 17, 6, 18, 7, 19, 8, 20, 9, 21, 10, 22, 11, 23], [3, 4, 2]);
      compare(permuted_4, target_4, 1e-3);

      // Permute axes to (2, 0, 1)
      const permuted_5 = x.permute(2, 0, 1);
      const target_5 = new Tensor("float32", [0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23], [4, 2, 3]);
      compare(permuted_5, target_5, 1e-3);

      // Permute axes to (2, 1, 0)
      const permuted_6 = x.permute(2, 1, 0);
      const target_6 = new Tensor("float32", [0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23], [4, 3, 2]);
      compare(permuted_6, target_6, 1e-3);
    });
  });

  describe("map", () => {
    it("should double", async () => {
      const original = new Tensor("float32", [1, 2, 3, 4, 5, 6], [2, 3]);
      const target = new Tensor("float32", [2, 4, 6, 8, 10, 12], [2, 3]);

      const doubled = original.map((x) => x * 2);
      compare(doubled, target, 1e-3);
    });
  });

  describe("mean", () => {
    it("should calculate mean", async () => {
      const t1 = new Tensor("float32", [1, 2, 3, 4, 5, 6], [2, 3, 1]);

      const target = new Tensor("float32", [3.5], []);

      const target0 = new Tensor("float32", [2.5, 3.5, 4.5], [3, 1]);
      const target1 = new Tensor("float32", [2, 5], [2, 1]);
      const target2 = new Tensor("float32", [1, 2, 3, 4, 5, 6], [2, 3]);

      let avg = mean(t1);
      compare(avg, target, 1e-3);

      let avg0 = mean(t1, 0);
      compare(avg0, target0, 1e-3);

      let avg1 = mean(t1, 1);
      compare(avg1, target1, 1e-3);

      let avg2 = mean(t1, 2);
      compare(avg2, target2, 1e-3);
    });
  });

  describe("layer_norm", () => {
    it("should calculate layer norm", async () => {
      const t1 = new Tensor("float32", [1, 2, 3, 4, 5, 6], [2, 3]);

      const target = new Tensor("float32", [-1.2247356176376343, 0.0, 1.2247356176376343, -1.2247357368469238, -1.1920928955078125e-7, 1.2247354984283447], [2, 3]);

      const norm = layer_norm(t1, [t1.dims.at(-1)]);
      compare(norm, target, 1e-3);
    });
  });
});
