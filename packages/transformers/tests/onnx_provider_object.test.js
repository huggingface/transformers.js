import { jest } from "@jest/globals";

import { env } from "../src/env.js";
import { AutoModelForCausalLM } from "../src/models/auto/modeling_auto.js";
import { PreTrainedModel } from "../src/models/modeling_utils.js";
import { DEFAULT_MODEL_OPTIONS, MAX_MODEL_LOAD_TIME } from "./init.js";

describe("ONNX provider objects", () => {
  it("configures the Transformers.js host before provider loading", async () => {
    const { OnnxInferenceProvider } = await import("@huggingface/transformers-onnx");
    const provider = OnnxInferenceProvider.from_modelId("test/provider-object");
    class TestModel extends PreTrainedModel {}
    TestModel._from_pretrained = jest.fn(async () => "loaded");

    await expect(TestModel.from_pretrained(provider, { config: { model_type: "custom" } })).resolves.toBe("loaded");

    expect(TestModel._from_pretrained).toHaveBeenCalledWith("test/provider-object", expect.objectContaining({ inferenceProvider: provider }));
    expect(typeof env.backends.onnx.setLogLevel).toBe("function");
  });

  it(
    "loads a real model through a provider object",
    async () => {
      const { OnnxInferenceProvider } = await import("@huggingface/transformers-onnx");
      const provider = OnnxInferenceProvider.from_modelId("hf-internal-testing/tiny-random-LlamaForCausalLM");
      const model = await AutoModelForCausalLM.from_pretrained(provider, DEFAULT_MODEL_OPTIONS);

      expect(model).toBeInstanceOf(PreTrainedModel);
      await model.dispose();
    },
    MAX_MODEL_LOAD_TIME,
  );
});
