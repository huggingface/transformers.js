import { jest } from "@jest/globals";

import { getModelId, isInferenceBackend, loadInferenceModel, normalizeInferenceModel, validateInferenceBackendTask, validateInferenceModelTask } from "../src/backends/inference.js";
import { OnnxInferenceProvider } from "@huggingface/transformers-onnx";
import { AutoModel } from "../src/models/auto/modeling_auto.js";
import { PreTrainedModel } from "../src/models/modeling_utils.js";
import { buildResourcePaths } from "../src/utils/hub.js";

describe("inference backends", () => {
  it("recognizes object and class backends", () => {
    const objectBackend = { modelId: "test/object", load() {} };
    class ClassBackend {
      static modelId = "test/class";
      static load() {}
    }

    expect(isInferenceBackend(objectBackend)).toBe(true);
    expect(isInferenceBackend(ClassBackend)).toBe(true);
    expect(getModelId(objectBackend)).toBe("test/object");
    expect(getModelId(ClassBackend)).toBe("test/class");
    expect(getModelId("test/string")).toBe("test/string");
  });

  it("normalizes a model with forward into a callable model", async () => {
    const implementation = {
      value: 42,
      async forward(inputs) {
        return { inputs, value: this.value };
      },
      async dispose() {},
    };
    const model = normalizeInferenceModel(implementation);

    await expect(model({ input_ids: "input" })).resolves.toEqual({
      inputs: { input_ids: "input" },
      value: 42,
    });
    expect(model.value).toBe(42);
  });

  it("passes shared loading options to a custom backend", async () => {
    const load = jest.fn(async () => ({
      async forward(inputs) {
        return inputs;
      },
      async dispose() {},
    }));
    const backend = { modelId: "test/model", load };
    const config = { model_type: "custom" };

    const model = await loadInferenceModel(backend, { config, dtype: "q4f16" });

    expect(load).toHaveBeenCalledWith({ config, dtype: "q4f16", modelId: "test/model" });
    expect(model.config).toBe(config);
  });

  it("rejects malformed artifact providers before backend loading", async () => {
    const backend = { modelId: "test/model", load: jest.fn() };

    await expect(loadInferenceModel(backend, { artifactProvider: { readJson() {} } })).rejects.toThrow("must implement `readJson()` and `openByteSource()`");
    expect(backend.load).not.toHaveBeenCalled();
  });

  it("uses declared task and loaded execution capabilities for setup validation", async () => {
    const backend = {
      modelId: "test/model",
      capabilities: { devices: ["webgpu"], dtypes: ["auto"], tasks: ["text-generation"] },
      load() {},
    };
    expect(() => validateInferenceBackendTask(backend, "feature-extraction")).toThrow('does not support the "feature-extraction" task');
    expect(() => validateInferenceBackendTask(backend, "text-generation")).not.toThrow();
    expect(() =>
      validateInferenceModelTask(
        {
          capabilities: { forward: { version: 1 } },
          async forward() {},
          async dispose() {},
        },
        "text-generation",
      ),
    ).toThrow("does not support causal text generation");
  });

  it("normalizes absent custom device and dtype options", async () => {
    const backend = {
      modelId: "test/model",
      load: jest.fn(async () => ({
        async forward(inputs) {
          return inputs;
        },
        async dispose() {},
      })),
    };

    await loadInferenceModel(backend, { device: null, dtype: null });

    expect(backend.load).toHaveBeenCalledWith(expect.objectContaining({ modelId: "test/model", device: undefined, dtype: undefined }));
  });

  it("allows custom backends through AutoModel.from_pretrained", async () => {
    const loaded = {
      async forward(inputs) {
        return inputs;
      },
      async dispose() {},
    };
    const backend = {
      modelId: "test/model",
      load: jest.fn(async () => loaded),
    };
    const signal = new AbortController().signal;
    const artifactProvider = { readJson() {}, openByteSource() {} };

    const model = await AutoModel.from_pretrained(backend, {
      config: { model_type: "custom" },
      device: "webgpu",
      signal,
      artifactProvider,
    });

    expect(backend.load).toHaveBeenCalledWith(expect.objectContaining({ modelId: "test/model", device: "webgpu", signal, artifactProvider }));
    await expect(model({ value: 1 })).resolves.toEqual({ value: 1 });
  });

  it("represents string IDs with the ONNX fallback backend", async () => {
    const modelClass = { _from_pretrained: jest.fn(async () => "model") };
    const backend = OnnxInferenceProvider.from_modelId("test/model");
    expect(backend.modelId).toBe("test/model");
    expect(isInferenceBackend(backend)).toBe(true);
    await expect(backend.load({ dtype: "q4", modelClass })).resolves.toBe("model");
    expect(modelClass._from_pretrained).toHaveBeenCalledWith("test/model", expect.objectContaining({ dtype: "q4", inferenceProvider: backend }));
  });

  it("resolves string model IDs through OnnxInferenceProvider.from_modelId", async () => {
    class TestModel extends PreTrainedModel {}
    TestModel._from_pretrained = jest.fn(async () => "loaded-model");
    const factory = jest.spyOn(OnnxInferenceProvider, "from_modelId");

    await expect(TestModel.from_pretrained("test/string-model")).resolves.toBe("loaded-model");

    expect(factory).toHaveBeenCalledWith("test/string-model");
    expect(TestModel._from_pretrained).toHaveBeenCalledWith("test/string-model", expect.objectContaining({ inferenceProvider: expect.any(OnnxInferenceProvider) }));
    factory.mockRestore();
  });

  it("uses backend model IDs for shared asset paths", () => {
    const backend = { modelId: "test/model", load() {} };
    const paths = buildResourcePaths(backend, "tokenizer.json");

    expect(paths.requestURL).toBe("test/model/tokenizer.json");
    expect(paths.remoteURL).toContain("/test/model/resolve/main/tokenizer.json");
  });
});
