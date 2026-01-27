import { getFiles, getProcessorFiles } from "../../src/configs.js";
import { getTokenizerFiles } from "../../src/tokenization_utils.js";
import { getModelFiles } from "../../src/models/modeling_utils.js";
import { env } from "../../src/env.js";
import { MAX_TEST_EXECUTION_TIME } from "../init.js";

// Initialise the testing environment
env.allowLocalModels = false;
env.useFSCache = false;

describe("File listing functions", () => {
  describe("getTokenizerFiles", () => {
    it("should require modelId parameter", async () => {
      await expect(getTokenizerFiles()).rejects.toThrow("modelId is required");
    });

    it(
      "should auto-detect tokenizer files for text models",
      async () => {
        const files = await getTokenizerFiles("Xenova/gpt2");
        expect(files).toEqual(["tokenizer.json", "tokenizer_config.json"]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "should auto-detect no tokenizer files for vision models",
      async () => {
        const files = await getTokenizerFiles("Xenova/detr-resnet-50");
        expect(files).toEqual([]);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });

  describe("getModelFiles", () => {
    it(
      "should return model files for decoder-only model (GPT-2)",
      async () => {
        const files = await getModelFiles("Xenova/gpt2");

        expect(Array.isArray(files)).toBe(true);
        expect(files.length).toBeGreaterThan(0);

        // Should include config.json
        expect(files).toContain("config.json");

        // Should include at least one .onnx file
        const hasOnnxFile = files.some((f) => f.endsWith(".onnx"));
        expect(hasOnnxFile).toBe(true);

        // Should include generation_config.json for decoder-only models
        expect(files).toContain("generation_config.json");
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "should return model files for encoder-decoder model (T5)",
      async () => {
        const files = await getModelFiles("Xenova/t5-small");

        expect(Array.isArray(files)).toBe(true);
        expect(files.length).toBeGreaterThan(0);

        // Should include config.json
        expect(files).toContain("config.json");

        // Should include encoder and decoder files
        const hasEncoderFile = files.some((f) => f.includes("encoder"));
        const hasDecoderFile = files.some((f) => f.includes("decoder"));
        expect(hasEncoderFile).toBe(true);
        expect(hasDecoderFile).toBe(true);

        // Should include generation_config.json
        expect(files).toContain("generation_config.json");
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "should return model files for encoder-only model (DistilBERT)",
      async () => {
        const files = await getModelFiles("Xenova/distilbert-base-uncased");

        expect(Array.isArray(files)).toBe(true);
        expect(files.length).toBeGreaterThan(0);

        // Should include config.json
        expect(files).toContain("config.json");

        // Should include model.onnx
        const hasModelFile = files.some((f) => f.includes("model.onnx"));
        expect(hasModelFile).toBe(true);

        // Should NOT include generation_config.json for encoder-only models
        expect(files).not.toContain("generation_config.json");
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "should detect external data files automatically",
      async () => {
        const files = await getModelFiles("onnx-community/all-MiniLM-L6-v2-ONNX");

        expect(Array.isArray(files)).toBe(true);
        expect(files.length).toBeGreaterThan(0);

        // Should include config.json
        expect(files).toContain("config.json");

        // Should include model.onnx
        const hasModelFile = files.some((f) => f.includes("model.onnx"));
        expect(hasModelFile).toBe(true);

        // Should include external data file (auto-detected from config)
        const hasExternalData = files.some((f) => f.includes("model.onnx_data"));
        expect(hasExternalData).toBe(true);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "should include subfolder in file paths",
      async () => {
        const files = await getModelFiles("Xenova/gpt2");

        // All .onnx files should be in the onnx/ subfolder
        const onnxFiles = files.filter((f) => f.endsWith(".onnx"));
        expect(onnxFiles.length).toBeGreaterThan(0);

        onnxFiles.forEach((file) => {
          expect(file).toMatch(/^onnx\//);
        });
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "should work with pre-loaded config",
      async () => {
        const { AutoConfig } = await import("../../src/transformers.js");
        const config = await AutoConfig.from_pretrained("Xenova/gpt2");
        const files = await getModelFiles("Xenova/gpt2", { config });

        expect(Array.isArray(files)).toBe(true);
        expect(files.length).toBeGreaterThan(0);
        expect(files).toContain("config.json");
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "should fail gracefully for non-existent model",
      async () => {
        await expect(getModelFiles("this-model/does-not-exist-12345")).rejects.toThrow();
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "should return unique file paths (no duplicates)",
      async () => {
        const files = await getModelFiles("Xenova/gpt2");
        const uniqueFiles = new Set(files);
        expect(uniqueFiles.size).toBe(files.length);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });

  describe("getFiles", () => {
    it(
      "should return combined tokenizer and model files",
      async () => {
        const files = await getFiles("Xenova/gpt2");

        expect(Array.isArray(files)).toBe(true);
        expect(files.length).toBeGreaterThan(0);

        // Should include tokenizer files
        expect(files).toContain("tokenizer.json");
        expect(files).toContain("tokenizer_config.json");

        // Should include model files
        expect(files).toContain("config.json");
        const hasOnnxFile = files.some((f) => f.endsWith(".onnx"));
        expect(hasOnnxFile).toBe(true);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "should have tokenizer files first, then model files",
      async () => {
        const files = await getFiles("Xenova/gpt2");

        // First two files should be tokenizer files
        expect(files[0]).toBe("tokenizer.json");
        expect(files[1]).toBe("tokenizer_config.json");

        // Remaining files should be model files
        expect(files.length).toBeGreaterThan(2);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "should work with encoder-decoder models",
      async () => {
        const files = await getFiles("Xenova/t5-small");

        expect(Array.isArray(files)).toBe(true);
        expect(files.length).toBeGreaterThan(0);

        // Should include tokenizer files
        expect(files).toContain("tokenizer.json");
        expect(files).toContain("tokenizer_config.json");

        // Should include model files
        expect(files).toContain("config.json");

        // Should include encoder and decoder
        const hasEncoderFile = files.some((f) => f.includes("encoder"));
        const hasDecoderFile = files.some((f) => f.includes("decoder"));
        expect(hasEncoderFile).toBe(true);
        expect(hasDecoderFile).toBe(true);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "should include external data files when present",
      async () => {
        const files = await getFiles("onnx-community/all-MiniLM-L6-v2-ONNX");

        expect(Array.isArray(files)).toBe(true);

        // Should include tokenizer files
        expect(files).toContain("tokenizer.json");
        expect(files).toContain("tokenizer_config.json");

        // Should include model files
        expect(files).toContain("config.json");

        // Should include external data (auto-detected)
        const hasExternalData = files.some((f) => f.includes("model.onnx_data"));
        expect(hasExternalData).toBe(true);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "should return at least 4 files for text models",
      async () => {
        const files = await getFiles("Xenova/gpt2");

        // Minimum for text models: tokenizer.json, tokenizer_config.json, model.onnx, config.json
        expect(files.length).toBeGreaterThanOrEqual(4);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "should work with pre-loaded config",
      async () => {
        const { AutoConfig } = await import("../../src/transformers.js");
        const config = await AutoConfig.from_pretrained("Xenova/gpt2");
        const files = await getFiles("Xenova/gpt2", { config });

        expect(Array.isArray(files)).toBe(true);
        expect(files).toContain("tokenizer.json");
        expect(files).toContain("config.json");
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "should return unique file paths (no duplicates)",
      async () => {
        const files = await getFiles("Xenova/gpt2");
        const uniqueFiles = new Set(files);
        expect(uniqueFiles.size).toBe(files.length);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });

  describe("Edge cases and error handling", () => {
    it(
      "should handle models with multiple external data chunks",
      async () => {
        // Note: We would need a model with multiple chunks to test this properly
        // For now, we just verify that the function doesn't crash
        const files = await getModelFiles("Xenova/gpt2");
        expect(Array.isArray(files)).toBe(true);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "should handle models without generation_config.json gracefully",
      async () => {
        const files = await getModelFiles("Xenova/distilbert-base-uncased");

        // Encoder-only models typically don't have generation_config.json
        expect(files).not.toContain("generation_config.json");

        // But should still have other files
        expect(files).toContain("config.json");
        expect(files.length).toBeGreaterThan(0);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });

  describe("Performance", () => {
    it(
      "should be fast when using cached config",
      async () => {
        const { AutoConfig } = await import("../../src/transformers.js");

        // First call - fetches config
        const config = await AutoConfig.from_pretrained("Xenova/gpt2");

        // Second call - uses cached config (should be faster)
        const startTime = Date.now();
        const files = await getModelFiles("Xenova/gpt2", { config });
        const duration = Date.now() - startTime;

        expect(files.length).toBeGreaterThan(0);

        // Should be very fast (< 100ms) when config is pre-loaded
        expect(duration).toBeLessThan(100);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });

  describe("Override parameters", () => {
    it(
      "should accept dtype override",
      async () => {
        // Test with fp32
        const files1 = await getModelFiles("onnx-community/gemma-3-270m-it-ONNX", {
          dtype: "fp32",
        });
        expect(files1.some((f) => f.includes("model.onnx"))).toBe(true);
        expect(files1.some((f) => f.includes("model_q4"))).toBe(false);

        // Test with q4
        const files2 = await getModelFiles("onnx-community/gemma-3-270m-it-ONNX", {
          dtype: "q4",
        });
        expect(files2.some((f) => f.includes("model_q4.onnx"))).toBe(true);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "should accept device override",
      async () => {
        // Device affects default dtype selection when dtype is not specified in config
        const filesWebGPU = await getModelFiles("Xenova/gpt2", {
          device: "webgpu",
        });
        const filesCPU = await getModelFiles("Xenova/gpt2", {
          device: "cpu",
        });

        // Both should have files
        expect(filesWebGPU.length).toBeGreaterThan(0);
        expect(filesCPU.length).toBeGreaterThan(0);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "should prefer override over config values",
      async () => {
        // Even if config has a dtype, override should take precedence
        const files = await getModelFiles("onnx-community/gemma-3-270m-it-ONNX", {
          dtype: "q4",
        });

        expect(files.some((f) => f.includes("model_q4.onnx"))).toBe(true);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });

  describe("Processor files", () => {
    it(
      "should auto-detect processor files for models that have them",
      async () => {
        const files = await getProcessorFiles("Xenova/whisper-tiny");
        expect(files).toContain("preprocessor_config.json");
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "should return empty array for models without processor",
      async () => {
        const files = await getProcessorFiles("Xenova/gpt2");
        expect(files).toEqual([]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "should auto-include processor files in getFiles for audio models",
      async () => {
        const files = await getFiles("Xenova/whisper-tiny");
        expect(files).toContain("preprocessor_config.json");
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "should work with audio models (Whisper)",
      async () => {
        const files = await getFiles("Xenova/whisper-tiny", {
          dtype: "uint8",
          device: "webgpu",
        });

        expect(files).toContain("preprocessor_config.json");
        expect(files).toContain("tokenizer.json");
        expect(files).toContain("config.json");
        expect(files.some((f) => f.includes("encoder_model_uint8.onnx"))).toBe(true);
        expect(files.some((f) => f.includes("decoder_model_merged_uint8.onnx"))).toBe(true);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });

  describe("Vision-only models (no tokenizer)", () => {
    it(
      "should auto-detect and exclude tokenizer files for vision models",
      async () => {
        const files = await getFiles("Xenova/detr-resnet-50");

        // DETR doesn't have tokenizer files, so they shouldn't be included
        expect(files).not.toContain("tokenizer.json");
        expect(files).not.toContain("tokenizer_config.json");
        expect(files).toContain("preprocessor_config.json");
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "should work with object detection models (DETR)",
      async () => {
        const files = await getFiles("Xenova/detr-resnet-50", {
          dtype: "uint8",
          device: "webgpu",
        });

        expect(files).not.toContain("tokenizer.json");
        expect(files).toContain("preprocessor_config.json");
        expect(files).toContain("config.json");
        expect(files.some((f) => f.includes("model_uint8.onnx"))).toBe(true);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "should auto-detect and include tokenizer files for text models",
      async () => {
        const files = await getFiles("Xenova/gpt2");

        // GPT-2 has tokenizer files, so they should be included
        expect(files).toContain("tokenizer.json");
        expect(files).toContain("tokenizer_config.json");
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
});
