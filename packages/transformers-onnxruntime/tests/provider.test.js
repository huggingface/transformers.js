import { jest } from "@jest/globals";
import { configureOnnxProviderHost, OnnxInferenceProvider } from "@huggingface/transformers-onnxruntime";

describe("OnnxInferenceProvider", () => {
  it("creates providers from model IDs", () => {
    const provider = OnnxInferenceProvider.from_modelId("onnx-community/test-model");

    expect(provider.modelId).toBe("onnx-community/test-model");
    expect(provider.providerType).toBe("onnx");
    expect(typeof provider.constructSessions).toBe("function");
  });

  it("owns architecture-to-artifact resolution", () => {
    expect(OnnxInferenceProvider.listModelArtifacts({
      modelType: "Seq2Seq",
      config: {},
      dtype: "q4",
      device: "wasm",
    })).toEqual([
      "config.json",
      "onnx/encoder_model_q4.onnx",
      "onnx/decoder_model_merged_q4.onnx",
      "generation_config.json",
    ]);

    const provider = OnnxInferenceProvider.from_modelId("onnx-community/test-model");
    expect(provider.getSessionConfig("DecoderOnly", {}, { model_file_name: "decoder" })).toEqual({
      sessions: { model: "decoder" },
      cache_sessions: { model: true },
      optional_configs: { generation_config: "generation_config.json" },
    });

    expect(OnnxInferenceProvider.listModelArtifacts({
      modelType: "Seq2Seq",
      config: {},
      dtype: { encoder_model: "fp16", decoder_model_merged: "q4" },
      device: "wasm",
      use_external_data_format: { "encoder_model_fp16.onnx": 1 },
    })).toEqual([
      "config.json",
      "onnx/encoder_model_fp16.onnx",
      "onnx/encoder_model_fp16.onnx_data",
      "onnx/decoder_model_merged_q4.onnx",
      "generation_config.json",
    ]);

    expect(OnnxInferenceProvider.filterModelArtifacts(
      ["onnx/embed_tokens.onnx", "onnx/decoder_model_merged.onnx", "onnx/encoder_model.onnx", "onnx/vision_encoder.onnx"],
      { modelType: "ImageTextToText", config: { is_encoder_decoder: true } },
    )).toEqual(["onnx/embed_tokens.onnx", "onnx/decoder_model_merged.onnx", "onnx/encoder_model.onnx"]);

    expect(OnnxInferenceProvider.listModelArtifacts({
      modelType: "EncoderOnly",
      config: {
        "transformers.js_config": {
          device: "webgpu",
          device_config: { webgpu: { dtype: "fp16", use_external_data_format: 1 } },
        },
      },
    })).toEqual(["config.json", "onnx/model_fp16.onnx", "onnx/model_fp16.onnx_data"]);
  });

  it("forwards the configured host fetch to model loading", async () => {
    const fetch = () => Promise.resolve(new Response());
    configureOnnxProviderHost({
      env: { backends: { onnx: {} }, fetch },
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
    const modelClass = { _from_pretrained: jest.fn(async () => "model") };

    await OnnxInferenceProvider.from_modelId("onnx-community/test-model").load({ modelClass });

    expect(modelClass._from_pretrained).toHaveBeenCalledWith(
      "onnx-community/test-model",
      expect.objectContaining({ fetch }),
    );
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
