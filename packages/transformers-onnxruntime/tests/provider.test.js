import { configureOnnxProviderHost, OnnxInferenceProvider } from "@huggingface/transformers-onnxruntime";

describe("OnnxInferenceProvider", () => {
  it("creates providers from model IDs", () => {
    const provider = OnnxInferenceProvider.from_modelId("onnx-community/test-model");

    expect(provider.modelId).toBe("onnx-community/test-model");
    expect(provider.providerType).toBe("onnx");
    expect(typeof provider.constructSessions).toBe("function");
  });

  it("uses the host external-data chunk limit", async () => {
    configureOnnxProviderHost({
      env: {
        backends: { onnx: {} },
        logLevel: 30,
        useWasmCache: false,
        fetch: globalThis.fetch,
      },
      apis: {
        IS_NODE_ENV: true,
        IS_WEB_ENV: false,
        IS_WEBGPU_AVAILABLE: false,
        IS_WEBNN_AVAILABLE: false,
        IS_DENO_WEB_RUNTIME: false,
        IS_SAFARI_BELOW_26: false,
        IS_SERVICE_WORKER_ENV: false,
        IS_CHROME_AVAILABLE: false,
      },
      logger: console,
      getModelFile: async () => new Uint8Array(),
      getCacheNames: () => new Set(),
      createBackendTensor: () => null,
      getBackendTensorStorage: () => null,
      maxExternalDataChunks: 2,
    });

    const provider = OnnxInferenceProvider.from_modelId("onnx-community/test-model");
    await expect(
      provider.getSession("model", {
        config: { "transformers.js_config": { use_external_data_format: 3 } },
      }),
    ).rejects.toThrow("exceeds the maximum allowed value (2)");
  });
});
