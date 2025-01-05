import { Tensor, cat, stack, layer_norm, ones_like, zeros_like, full_like, rand, std_mean } from "../../src/transformers.js";
import { init } from "../init.js";
import { compare } from "../test_utils.js";

init();

describe("Tensor operations", () => {
  describe("cat", () => {
    it("should concatenate on dim=0", () => {
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

    it("should concatenate on dim=1", () => {
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

    it("should concatenate on dim=-2", () => {
      const t1 = new Tensor("float32", [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16], [2, 1, 3, 2]);
      const t2 = new Tensor("float32", [7, 8, 9, 10, 17, 18, 19, 20], [2, 1, 2, 2]);

      const target = new Tensor("float32", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], [2, 1, 5, 2]);

      const concatenated = cat([t1, t2], -2);

      compare(concatenated, target, 1e-3);
    });

    // TODO add tests for errors
  });

  describe("slice", () => {
    it("should return a given row dim", () => {
      const t1 = new Tensor("float32", [1, 2, 3, 4, 5, 6], [3, 2]);
      const t2 = t1.slice(1);
      const target = new Tensor("float32", [3, 4], [2]);

      compare(t2, target);
    });

    it("should return a range of rows", () => {
      const t1 = new Tensor("float32", [1, 2, 3, 4, 5, 6], [3, 2]);
      const t2 = t1.slice([1, 3]);
      const target = new Tensor("float32", [3, 4, 5, 6], [2, 2]);

      compare(t2, target);
    });

    it("should return a crop", () => {
      const t1 = new Tensor(
        "float32",
        Array.from({ length: 28 }, (_, i) => i + 1),
        [4, 7],
      );
      const t2 = t1.slice([1, -1], [1, -1]);

      const target = new Tensor("float32", [9, 10, 11, 12, 13, 16, 17, 18, 19, 20], [2, 5]);

      compare(t2, target);
    });
  });

  describe("stack", () => {
    const t1 = new Tensor("float32", [0, 1, 2, 3, 4, 5], [1, 3, 2]);

    it("should stack on dim=0", () => {
      const target1 = new Tensor("float32", [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], [2, 1, 3, 2]);
      const target2 = new Tensor("float32", [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], [3, 1, 3, 2]);

      // 2 tensors
      const stacked1 = stack([t1, t1], 0);
      compare(stacked1, target1, 1e-3);

      // 3 tensors
      const stacked2 = stack([t1, t1, t1], 0);
      compare(stacked2, target2, 1e-3);
    });

    it("should stack on dim=1", () => {
      const target1 = new Tensor("float32", [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], [1, 2, 3, 2]);
      const target2 = new Tensor("float32", [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], [1, 3, 3, 2]);

      // 2 tensors
      const stacked1 = stack([t1, t1], 1);
      compare(stacked1, target1, 1e-3);

      // 3 tensors
      const stacked2 = stack([t1, t1, t1], 1);
      compare(stacked2, target2, 1e-3);
    });

    it("should stack on dim=-1", () => {
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
    it("should permute", () => {
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
    it("should double", () => {
      const original = new Tensor("float32", [1, 2, 3, 4, 5, 6], [2, 3]);
      const target = new Tensor("float32", [2, 4, 6, 8, 10, 12], [2, 3]);

      const doubled = original.map((x) => x * 2);
      compare(doubled, target, 1e-3);
    });
  });

  describe("mean", () => {
    const t1 = new Tensor("float32", [1, 2, 3, 4, 5, 6], [2, 3, 1]);
    it("should calculate mean over the entire tensor", () => {
      const target = new Tensor("float32", [3.5], []);
      compare(t1.mean(), target, 1e-3);
    });

    it("should calculate mean over dimension 0", () => {
      const target0 = new Tensor("float32", [2.5, 3.5, 4.5], [3, 1]);
      compare(t1.mean(0), target0, 1e-3);
    });

    it("should calculate mean over dimension 1", () => {
      const target1 = new Tensor("float32", [2, 5], [2, 1]);
      compare(t1.mean(1), target1, 1e-3);
    });

    it("should calculate mean over dimension -1", () => {
      const target2 = new Tensor("float32", [1, 2, 3, 4, 5, 6], [2, 3]);
      compare(t1.mean(-1), target2, 1e-3);
    });
  });

  describe("std_mean", () => {
    it("should return std_mean for the entire tensor", () => {
      const t = new Tensor("float32", [1, 2, 3, 4, 5, 6], [2, 3]);
      const [stdVal, meanVal] = std_mean(t);
      compare(stdVal, new Tensor("float32", [1.8708287477493286], []), 1e-3);
      compare(meanVal, new Tensor("float32", [3.5], []), 1e-3);
    });
  });

  describe("min", () => {
    it("should return the minimum over the entire tensor", () => {
      const t1 = new Tensor("float32", [3, -2, 5, 0], [2, 2]);
      const target = new Tensor("float32", [-2], []);
      const result = t1.min();
      compare(result, target, 1e-3);
    });

    it("should return the minimum over dimension 1", () => {
      const t2 = new Tensor("float32", [4, 2, -1, 0, 6, 5], [3, 2]);
      const target = new Tensor("float32", [2, -1, 5], [3]);
      const result = t2.min(1);
      compare(result, target, 1e-3);
    });
  });

  describe("max", () => {
    it("should return the maximum over the entire tensor", () => {
      const t1 = new Tensor("float32", [3, 10, -2, 7], [2, 2]);
      const target = new Tensor("float32", [10], []);
      const result = t1.max();
      compare(result, target, 1e-3);
    });

    it("should return the maximum over dimension 0", () => {
      const t2 = new Tensor("float32", [1, 2, 4, 5, 9, 3], [3, 2]);
      const target = new Tensor("float32", [9, 5], [2]);
      const result = t2.max(0);
      compare(result, target, 1e-3);
    });
  });

  describe("sum", () => {
    it("should calculate sum over entire tensor", () => {
      const t1 = new Tensor("float32", [1, 2, 3, 4, 5, 6], [2, 3]);
      const target = new Tensor("float32", [21], []);
      const result = t1.sum();
      compare(result, target, 1e-3);
    });

    it("should calculate sum over dimension 0", () => {
      const t1 = new Tensor("float32", [1, 2, 3, 4, 5, 6], [2, 3]);
      const target = new Tensor("float32", [5, 7, 9], [3]);
      const result = t1.sum(0);
      compare(result, target, 1e-3);
    });

    it("should calculate sum over dimension 1", () => {
      const t1 = new Tensor("float32", [1, 2, 3, 4, 5, 6], [2, 3]);
      const target = new Tensor("float32", [6, 15], [2]);
      const result = t1.sum(1);
      compare(result, target, 1e-3);
    });
  });

  describe("norm", () => {
    it("should calculate L2 norm over entire tensor", () => {
      const t1 = new Tensor("float32", [3, 4], [2]);
      const target = new Tensor("float32", [5], []);
      const result = t1.norm();
      compare(result, target, 1e-3);
    });

    it("should calculate L2 norm over dimension 0", () => {
      const t1 = new Tensor("float32", [3, 4, 6, 8], [2, 2]);
      const target = new Tensor("float32", [6.7082, 8.9443], [2]);
      const result = t1.norm(2, 0);
      compare(result, target, 1e-2);
    });
  });

  describe("normalize", () => {
    it("should normalize a vector correctly", () => {
      const t1 = new Tensor("float32", [3, 4], [1, 2]);
      const target = new Tensor("float32", [0.6, 0.8], [1, 2]);
      const normalized = t1.normalize();
      compare(normalized, target, 1e-3);
    });

    it("should normalize along dimension", () => {
      const t1 = new Tensor("float32", [1, 2, 2, 3], [2, 2]);
      const target = new Tensor("float32", [0.4472, 0.8944, 0.5547, 0.8321], [2, 2]);
      const normalized = t1.normalize();
      compare(normalized, target, 1e-3);
    });
  });

  describe("layer_norm", () => {
    it("should calculate layer norm", () => {
      const t1 = new Tensor("float32", [1, 2, 3, 4, 5, 6], [2, 3]);

      const target = new Tensor("float32", [-1.2247356176376343, 0.0, 1.2247356176376343, -1.2247357368469238, -1.1920928955078125e-7, 1.2247354984283447], [2, 3]);

      const norm = layer_norm(t1, [t1.dims.at(-1)]);
      compare(norm, target, 1e-3);
    });
  });

  describe("sigmoid", () => {
    it("should apply the sigmoid function to each element in the tensor", () => {
      const t1 = new Tensor("float32", [0, 1, -1, 5, -5], [5]);
      const target = new Tensor("float32", [0.5, 1 / (1 + Math.exp(-1)), 1 / (1 + Math.exp(1)), 1 / (1 + Math.exp(-5)), 1 / (1 + Math.exp(5))], [5]);

      const result = t1.sigmoid();
      compare(result, target, 1e-3);
    });
  });

  describe("tolist", () => {
    it("should return nested arrays for a 2D tensor", () => {
      const t1 = new Tensor("float32", [1, 2, 3, 4], [2, 2]);
      const arr = t1.tolist();
      compare(arr, [
        [1, 2],
        [3, 4],
      ]);
    });
  });

  describe("mul", () => {
    it("should multiply constant", () => {
      const t1 = new Tensor("float32", [1, 2, 3, 4], [2, 2]);
      const target = new Tensor("float32", [2, 4, 6, 8], [2, 2]);

      const result = t1.mul(2);
      compare(result, target, 1e-3);
    });
  });

  describe("div", () => {
    it("should divide constant", () => {
      const t1 = new Tensor("float32", [1, 2, 3, 4], [2, 2]);
      const target = new Tensor("float32", [0.5, 1, 1.5, 2], [2, 2]);

      const result = t1.div(2);
      compare(result, target, 1e-3);
    });
  });

  describe("add", () => {
    it("should add constant", () => {
      const t1 = new Tensor("float32", [1, 2, 3, 4], [2, 2]);
      const target = new Tensor("float32", [3, 4, 5, 6], [2, 2]);

      const result = t1.add(2);
      compare(result, target, 1e-3);
    });
  });

  describe("sub", () => {
    it("should subtract constant", () => {
      const t1 = new Tensor("float32", [1, 2, 3, 4], [2, 2]);
      const target = new Tensor("float32", [-1, 0, 1, 2], [2, 2]);

      const result = t1.sub(2);
      compare(result, target, 1e-3);
    });
  });
  describe("gt", () => {
    it("should perform element-wise greater than comparison with a scalar", () => {
      const t1 = new Tensor("float32", [1, 5, 3, 7], [4]);
      const target = new Tensor("bool", [0, 1, 0, 1], [4]);
      const result = t1.gt(4);
      compare(result, target, 1e-3);
    });
  });

  describe("lt", () => {
    it("should perform element-wise less than comparison with a scalar", () => {
      const t1 = new Tensor("float32", [1, 5, 3, 7], [4]);
      const target = new Tensor("bool", [1, 0, 1, 0], [4]);
      const result = t1.lt(4);
      compare(result, target, 1e-3);
    });
  });

  describe("squeeze", () => {
    it("should remove all dimensions of size 1", () => {
      const t1 = new Tensor("float32", [1, 2, 3, 4], [1, 4]);
      const target = new Tensor("float32", [1, 2, 3, 4], [4]);

      const result = t1.squeeze();
      compare(result, target, 1e-3);
    });
    it("should remove a specified dimension", () => {
      const t1 = new Tensor("float32", [1, 2, 3, 4], [1, 1, 2, 2]);
      const result = t1.squeeze(1);
      const target = new Tensor("float32", [1, 2, 3, 4], [1, 2, 2]);
      compare(result, target, 1e-3);
    });
    it("should remove multiple dimensions", () => {
      const t1 = new Tensor("float32", [1, 2, 3, 4], [1, 1, 2, 1, 2]);
      const result = t1.squeeze([0, 3]);
      const target = new Tensor("float32", [1, 2, 3, 4], [1, 2, 2]);
      compare(result, target, 1e-3);
    });
  });

  describe("unsqueeze", () => {
    it("should add a dimension at the specified axis", () => {
      const t1 = new Tensor("float32", [1, 2, 3, 4], [4]);
      const target = new Tensor("float32", [1, 2, 3, 4], [1, 4]);

      const result = t1.unsqueeze(0);
      compare(result, target, 1e-3);
    });
  });

  describe("flatten", () => {
    it("should flatten a 2D tensor into 1D by default", () => {
      const t1 = new Tensor("float32", [1, 2, 3, 4, 5, 6], [2, 3]);
      const target = new Tensor("float32", [1, 2, 3, 4, 5, 6], [6]);

      const result = t1.flatten();
      compare(result, target, 1e-3);
    });
  });

  describe("neg", () => {
    it("should compute the negative of each element in the tensor", () => {
      const t1 = new Tensor("float32", [1, -2, 0, 3], [4]);
      const target = new Tensor("float32", [-1, 2, -0, -3], [4]);

      const result = t1.neg();
      compare(result, target, 1e-3);
    });
  });

  describe("view", () => {
    it("should reshape the tensor to the specified dimensions", () => {
      const t1 = new Tensor("float32", [1, 2, 3, 4, 5, 6], [2, 3]);
      const target = new Tensor("float32", [1, 2, 3, 4, 5, 6], [3, 2]);

      const result = t1.view(3, 2);
      compare(result, target, 1e-3);
    });

    it("should reshape the tensor with an inferred dimension (-1)", () => {
      const t1 = new Tensor("float32", [1, 2, 3, 4, 5, 6], [2, 3]);
      const target = new Tensor("float32", [1, 2, 3, 4, 5, 6], [1, 6]);

      const result = t1.view(1, -1);
      compare(result, target, 1e-3);
    });

    it("should throw if multiple inferred dimensions are used", () => {
      const t1 = new Tensor("float32", [1, 2, 3, 4, 5, 6], [2, 3]);
      expect(() => t1.view(-1, -1)).toThrow();
    });
  });

  describe("clamp", () => {
    it("should clamp values between min and max", () => {
      const t1 = new Tensor("float32", [-2, -1, 0, 1, 2, 3], [6]);
      const target = new Tensor("float32", [-1, -1, 0, 1, 2, 2], [6]);

      const result = t1.clamp(-1, 2);
      compare(result, target, 1e-3);
    });
  });

  describe("round", () => {
    it("should round elements to the nearest integer", () => {
      const t1 = new Tensor("float32", [0.1, 1.4, 2.5, 3.9, -1.2], [5]);
      const target = new Tensor("float32", [0, 1, 3, 4, -1], [5]);

      const result = t1.round();
      compare(result, target, 1e-3);
    });
  });

  describe("ones_like", () => {
    it("should create a tensor of all ones with the same shape as the input", () => {
      const t1 = new Tensor("float32", [1, 2, 3, 4], [2, 2]);
      const result = ones_like(t1);
      const target = new Tensor("int64", [1n, 1n, 1n, 1n], [2, 2]);
      compare(result, target, 1e-3);
    });
  });

  describe("zeros_like", () => {
    it("should create a tensor of all zeros with the same shape as the input", () => {
      const t1 = new Tensor("float32", [1, 2, 3, 4], [2, 2]);
      const result = zeros_like(t1);
      const target = new Tensor("int64", [0n, 0n, 0n, 0n], [2, 2]);
      compare(result, target, 1e-3);
    });
  });

  describe("full_like", () => {
    it("should create a tensor filled with a number, matching the shape of the original", () => {
      const t1 = new Tensor("float32", [1, 2, 3, 4], [2, 2]);
      const result = full_like(t1, 10);
      const target = new Tensor("float32", [10, 10, 10, 10], [2, 2]);
      compare(result, target, 1e-3);
    });
    it("should create a boolean tensor with the same shape", () => {
      const t2 = new Tensor("bool", [true, false], [2]);
      const result = full_like(t2, true);
      const target = new Tensor("bool", [true, true], [2]);
      compare(result, target, 1e-3);
    });

    it("should create a bigint tensor with the same shape", () => {
      const t3 = new Tensor("int64", [1n, 2n], [2]);
      const result = full_like(t3, 123n);
      const target = new Tensor("int64", [123n, 123n], [2]);
      compare(result, target, 1e-3);
    });
  });

  describe("rand", () => {
    it("should create a tensor of random values between 0 and 1 with the given shape", () => {
      const shape = [2, 2];
      const random = rand(shape);
      expect(random.type).toBe("float32");
      expect(random.dims).toEqual(shape);
      random.data.forEach((val) => {
        expect(val).toBeGreaterThanOrEqual(0);
        expect(val).toBeLessThan(1);
      });
    });
  });

  describe("to", () => {
    it("float32 to int32 (number to number)", () => {
      const t1 = new Tensor("float32", [1, 2, 3, 4, 5, 6], [2, 3]);

      const target = new Tensor("int32", [1, 2, 3, 4, 5, 6], [2, 3]);

      const t2 = t1.to("int32");
      compare(t2, target);
    });
    it("float32 to int64 (number to bigint)", () => {
      const t1 = new Tensor("float32", [1, 2, 3, 4, 5, 6], [2, 3]);

      const target = new Tensor("int64", [1n, 2n, 3n, 4n, 5n, 6n], [2, 3]);

      const t2 = t1.to("int64");
      compare(t2, target);
    });
    it("int64 to float32 (bigint to number)", () => {
      const t1 = new Tensor("int64", [1n, 2n, 3n, 4n, 5n, 6n], [2, 3]);

      const target = new Tensor("float32", [1, 2, 3, 4, 5, 6], [2, 3]);

      const t2 = t1.to("float32");
      compare(t2, target);
    });
    it("int32 to uint32", () => {
      const t1 = new Tensor("int32", [-1, 2, -3, 4, -5, 6], [2, 3]);

      const target = new Tensor("uint32", [4294967295, 2, 4294967293, 4, 4294967291, 6], [2, 3]);

      const t2 = t1.to("uint32");
      compare(t2, target);
    });
    it("int16 to int8 (overflow)", () => {
      const t1 = new Tensor("int16", [0, 1, 128, 256, 257, 512], [2, 3]);

      const target = new Tensor("int8", [0, 1, -128, 0, 1, 0], [2, 3]);

      const t2 = t1.to("int8");
      compare(t2, target);
    });
  });
});
