import { OnnxInferenceProvider } from "@huggingface/transformers-onnx";

describe("OnnxInferenceProvider", () => {
  it("creates providers from model IDs", () => {
    const provider = OnnxInferenceProvider.from_modelId("onnx-community/test-model");

    expect(provider.modelId).toBe("onnx-community/test-model");
    expect(provider.providerType).toBe("onnx");
    expect(typeof provider.constructSessions).toBe("function");
  });
});
