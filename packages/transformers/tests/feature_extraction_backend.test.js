import { jest } from "@jest/globals";

import { FeatureExtractionPipeline } from "../src/pipelines/feature-extraction.js";
import { Tensor, TensorOpRegistry } from "../src/transformers.js";

describe("device-resident feature extraction", () => {
  it("does not fall through to default operations for custom backend tensors", async () => {
    const tensor = Tensor.fromBackendStorage({
      backend: "unregistered-webgpu",
      handle: {},
      type: "float32",
      dims: [1],
      size: 1,
      location: "gpu-buffer",
      get data() {
        throw new Error("Unexpected readback.");
      },
      dispose() {},
    });

    await expect(TensorOpRegistry.resolve("top_k", [tensor])).rejects.toThrow('Tensor operation "top_k" for backend "unregistered-webgpu" requires a registered implementation.');
    tensor.dispose();
  });

  it("dispatches pooling and normalization to the tensor backend without reading CPU data", async () => {
    const disposals = [];
    const createDeviceTensor = (stage, dims) => {
      const dispose = jest.fn();
      disposals.push(dispose);
      return Tensor.fromBackendStorage({
        backend: "test-webgpu",
        handle: { stage },
        type: "float32",
        dims,
        get data() {
          throw new Error(`Unexpected ${stage} readback.`);
        },
        size: dims.reduce((product, dimension) => product * dimension, 1),
        location: "gpu-buffer",
        dispose,
      });
    };

    const operations = {
      mean_pooling: jest.fn(async (input, attentionMask) => {
        expect(input.getBackendStorage("test-webgpu").handle.stage).toBe("model-output");
        expect(attentionMask.backend).toBe("cpu");
        return createDeviceTensor("pooled", [1, 3]);
      }),
      normalize: jest.fn(async (input, p, dim) => {
        expect(input.getBackendStorage("test-webgpu").handle.stage).toBe("pooled");
        expect([p, dim]).toEqual([2, -1]);
        return createDeviceTensor("normalized", [1, 3]);
      }),
    };
    TensorOpRegistry.register("test-webgpu", operations);

    try {
      const tokenizer = () => ({ attention_mask: new Tensor("int64", [1n, 1n], [1, 2]) });
      const model = jest.fn(async () => ({ last_hidden_state: createDeviceTensor("model-output", [1, 2, 3]) }));
      const extractor = new FeatureExtractionPipeline({ task: "feature-extraction", model, tokenizer });

      const result = await extractor._call("device resident", { pooling: "mean", normalize: true });

      expect(result.backend).toBe("test-webgpu");
      expect(result.location).toBe("gpu-buffer");
      expect(result.getBackendStorage("test-webgpu").handle.stage).toBe("normalized");
      expect(operations.mean_pooling).toHaveBeenCalledTimes(1);
      expect(operations.normalize).toHaveBeenCalledTimes(1);
      expect(disposals[0]).toHaveBeenCalledTimes(1);
      expect(disposals[1]).toHaveBeenCalledTimes(1);
      expect(disposals[2]).not.toHaveBeenCalled();
      result.dispose();
      expect(disposals[2]).toHaveBeenCalledTimes(1);
    } finally {
      TensorOpRegistry.unregister("test-webgpu", operations);
    }
  });
});
