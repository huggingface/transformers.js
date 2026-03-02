import { ModelRegistry } from "../../src/transformers.js";
import { getModelFile } from "../../src/utils/hub.js";

import { MAX_TEST_EXECUTION_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";

const LLAMA_MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM";
const BERT_MODEL_ID = "hf-internal-testing/tiny-random-BertModel";
const VIT_MODEL_ID = "hf-internal-testing/tiny-random-vit";

// Dedicated model IDs for cache clearing tests to avoid interference with other parallel tests.
// These must NOT be used in any other test file.
const CLEAR_CACHE_MODEL_ID = "onnx-internal-testing/tiny-random-BertModel-ONNX";
const CLEAR_PIPELINE_CACHE_MODEL_ID = "onnx-internal-testing/tiny-random-Qwen3ForCausalLM";

describe("Cache", () => {
  describe("ModelRegistry", () => {
    describe("get_files", () => {
      it(
        "should return files for a decoder-only model",
        async () => {
          const files = await ModelRegistry.get_files(LLAMA_MODEL_ID, DEFAULT_MODEL_OPTIONS);
          expect(Array.isArray(files)).toBe(true);
          expect(files.length).toBeGreaterThan(0);
          expect(files).toContain("config.json");
          expect(files).toContain("generation_config.json");
          expect(files.some((f) => f.startsWith("onnx/") && f.endsWith(".onnx"))).toBe(true);
          expect(files).toContain("tokenizer.json");
          expect(files).toContain("tokenizer_config.json");
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "should return files for an encoder-only model",
        async () => {
          const files = await ModelRegistry.get_files(BERT_MODEL_ID, DEFAULT_MODEL_OPTIONS);
          expect(Array.isArray(files)).toBe(true);
          expect(files.length).toBeGreaterThan(0);
          expect(files).toContain("config.json");
          expect(files.some((f) => f.startsWith("onnx/") && f.endsWith(".onnx"))).toBe(true);
          expect(files).toContain("tokenizer.json");
          expect(files).toContain("tokenizer_config.json");
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    describe("get_model_files", () => {
      it(
        "should return model files for a decoder-only model",
        async () => {
          const files = await ModelRegistry.get_model_files(LLAMA_MODEL_ID, DEFAULT_MODEL_OPTIONS);
          expect(Array.isArray(files)).toBe(true);
          expect(files).toContain("config.json");
          expect(files).toContain("generation_config.json");
          expect(files.some((f) => f.startsWith("onnx/") && f.endsWith(".onnx"))).toBe(true);
          // Should not include tokenizer files
          expect(files).not.toContain("tokenizer.json");
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "should return model files for an encoder-only model",
        async () => {
          const files = await ModelRegistry.get_model_files(BERT_MODEL_ID, DEFAULT_MODEL_OPTIONS);
          expect(Array.isArray(files)).toBe(true);
          expect(files).toContain("config.json");
          expect(files.some((f) => f.startsWith("onnx/") && f.endsWith(".onnx"))).toBe(true);
          // Encoder-only models should not have generation_config.json
          expect(files).not.toContain("generation_config.json");
          // Should not include tokenizer files
          expect(files).not.toContain("tokenizer.json");
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "should use model_file_name when provided",
        async () => {
          const files = await ModelRegistry.get_model_files(BERT_MODEL_ID, {
            ...DEFAULT_MODEL_OPTIONS,
            model_file_name: "custom_model",
          });
          expect(files).toContain("config.json");
          // Should use custom model file name
          expect(files.some((f) => f.includes("custom_model") && f.endsWith(".onnx"))).toBe(true);
          // Should NOT contain the default 'model' name
          expect(files.some((f) => f === "onnx/model.onnx")).toBe(false);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    describe("get_tokenizer_files", () => {
      it(
        "should return tokenizer files for a decoder-only model",
        async () => {
          const files = await ModelRegistry.get_tokenizer_files(LLAMA_MODEL_ID);
          expect(files).toEqual(["tokenizer.json", "tokenizer_config.json"]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "should return tokenizer files for an encoder-only model",
        async () => {
          const files = await ModelRegistry.get_tokenizer_files(BERT_MODEL_ID);
          expect(files).toEqual(["tokenizer.json", "tokenizer_config.json"]);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    describe("get_processor_files", () => {
      it(
        "should return empty array for text-only models",
        async () => {
          const llamaFiles = await ModelRegistry.get_processor_files(LLAMA_MODEL_ID);
          expect(llamaFiles).toEqual([]);

          const bertFiles = await ModelRegistry.get_processor_files(BERT_MODEL_ID);
          expect(bertFiles).toEqual([]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "should return processor files for a vision model",
        async () => {
          const files = await ModelRegistry.get_processor_files(VIT_MODEL_ID);
          expect(files).toContain("preprocessor_config.json");
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    describe("get_pipeline_files", () => {
      it(
        "should return files for text-generation pipeline",
        async () => {
          const files = await ModelRegistry.get_pipeline_files("text-generation", LLAMA_MODEL_ID, DEFAULT_MODEL_OPTIONS);
          expect(Array.isArray(files)).toBe(true);
          expect(files).toContain("config.json");
          expect(files).toContain("generation_config.json");
          expect(files.some((f) => f.startsWith("onnx/") && f.endsWith(".onnx"))).toBe(true);
          expect(files).toContain("tokenizer.json");
          expect(files).toContain("tokenizer_config.json");
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "should return files for feature-extraction pipeline",
        async () => {
          const files = await ModelRegistry.get_pipeline_files("feature-extraction", BERT_MODEL_ID, DEFAULT_MODEL_OPTIONS);
          expect(Array.isArray(files)).toBe(true);
          expect(files).toContain("config.json");
          expect(files.some((f) => f.startsWith("onnx/") && f.endsWith(".onnx"))).toBe(true);
          expect(files).toContain("tokenizer.json");
          expect(files).toContain("tokenizer_config.json");
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "should resolve task aliases",
        async () => {
          const files = await ModelRegistry.get_pipeline_files("sentiment-analysis", BERT_MODEL_ID, DEFAULT_MODEL_OPTIONS);
          expect(Array.isArray(files)).toBe(true);
          expect(files).toContain("config.json");
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "should throw for unsupported pipeline task",
        async () => {
          await expect(ModelRegistry.get_pipeline_files("invalid-nonexistent-task", BERT_MODEL_ID, DEFAULT_MODEL_OPTIONS)).rejects.toThrow("Unsupported pipeline task");
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "should not include processor files for text-only tasks",
        async () => {
          const files = await ModelRegistry.get_pipeline_files("text-generation", LLAMA_MODEL_ID, DEFAULT_MODEL_OPTIONS);
          // text-generation tasks don't use a processor, so no preprocessor_config.json
          expect(files).not.toContain("preprocessor_config.json");
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "should include processor files for image tasks",
        async () => {
          const files = await ModelRegistry.get_pipeline_files("image-classification", VIT_MODEL_ID, DEFAULT_MODEL_OPTIONS);
          expect(files).toContain("preprocessor_config.json");
          // image-classification doesn't use a tokenizer
          expect(files).not.toContain("tokenizer.json");
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    describe("is_cached", () => {
      it(
        "should return cache status with correct shape",
        async () => {
          const status = await ModelRegistry.is_cached(BERT_MODEL_ID, DEFAULT_MODEL_OPTIONS);
          expect(status).toHaveProperty("allCached");
          expect(typeof status.allCached).toBe("boolean");
          expect(status).toHaveProperty("files");
          expect(Array.isArray(status.files)).toBe(true);
          expect(status.files.length).toBeGreaterThan(0);
          for (const entry of status.files) {
            expect(entry).toHaveProperty("file");
            expect(typeof entry.file).toBe("string");
            expect(entry).toHaveProperty("cached");
            expect(typeof entry.cached).toBe("boolean");
          }
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    describe("is_pipeline_cached", () => {
      it(
        "should return cache status for text-generation pipeline",
        async () => {
          const status = await ModelRegistry.is_pipeline_cached("text-generation", LLAMA_MODEL_ID, DEFAULT_MODEL_OPTIONS);
          expect(status).toHaveProperty("allCached");
          expect(typeof status.allCached).toBe("boolean");
          expect(status).toHaveProperty("files");
          expect(Array.isArray(status.files)).toBe(true);
          expect(status.files.length).toBeGreaterThan(0);
          for (const entry of status.files) {
            expect(entry).toHaveProperty("file");
            expect(entry).toHaveProperty("cached");
          }
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    describe("get_file_metadata", () => {
      it(
        "should return metadata for an existing file",
        async () => {
          const metadata = await ModelRegistry.get_file_metadata(BERT_MODEL_ID, "config.json");
          expect(metadata).toHaveProperty("exists", true);
          expect(metadata).toHaveProperty("size");
          expect(typeof metadata.size).toBe("number");
          expect(metadata.size).toBeGreaterThan(0);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "should return exists=false for a non-existent file",
        async () => {
          const metadata = await ModelRegistry.get_file_metadata(BERT_MODEL_ID, "nonexistent_file.bin");
          expect(metadata).toHaveProperty("exists", false);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    describe("clear_cache", () => {
      it(
        "should clear cached files and report results",
        async () => {
          // Step 1: Pre-cache config.json by downloading it
          await getModelFile(CLEAR_CACHE_MODEL_ID, "config.json", true, {});

          // Step 2: Clear the cache
          const result = await ModelRegistry.clear_cache(CLEAR_CACHE_MODEL_ID, DEFAULT_MODEL_OPTIONS);

          // Step 3: Verify response shape
          expect(result).toHaveProperty("filesDeleted");
          expect(typeof result.filesDeleted).toBe("number");
          expect(result).toHaveProperty("filesCached");
          expect(typeof result.filesCached).toBe("number");
          expect(result).toHaveProperty("files");
          expect(Array.isArray(result.files)).toBe(true);
          for (const entry of result.files) {
            expect(entry).toHaveProperty("file");
            expect(typeof entry.file).toBe("string");
            expect(entry).toHaveProperty("deleted");
            expect(typeof entry.deleted).toBe("boolean");
            expect(entry).toHaveProperty("wasCached");
            expect(typeof entry.wasCached).toBe("boolean");
          }

          // Step 4: config.json should have been cached and deleted
          // (it was pre-cached in Step 1)
          const configEntry = result.files.find((f) => f.file === "config.json");
          expect(configEntry?.wasCached).toBe(true);
          expect(configEntry?.deleted).toBe(true);
          expect(result.filesDeleted).toBeGreaterThan(0);

          // NOTE: We don't re-check is_cached here because it internally calls
          // get_model_files() -> AutoConfig.from_pretrained(), which re-downloads
          // config.json and re-populates the cache as a side effect.
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    describe("clear_pipeline_cache", () => {
      it(
        "should clear cached pipeline files and report results",
        async () => {
          // Step 1: Pre-cache config.json by downloading it
          await getModelFile(CLEAR_PIPELINE_CACHE_MODEL_ID, "config.json", true, {});

          // Step 2: Clear the pipeline cache
          const result = await ModelRegistry.clear_pipeline_cache("text-generation", CLEAR_PIPELINE_CACHE_MODEL_ID, DEFAULT_MODEL_OPTIONS);

          // Step 3: Verify response shape
          expect(result).toHaveProperty("filesDeleted");
          expect(typeof result.filesDeleted).toBe("number");
          expect(result).toHaveProperty("filesCached");
          expect(typeof result.filesCached).toBe("number");
          expect(result).toHaveProperty("files");
          expect(Array.isArray(result.files)).toBe(true);
          for (const entry of result.files) {
            expect(entry).toHaveProperty("file");
            expect(entry).toHaveProperty("deleted");
            expect(entry).toHaveProperty("wasCached");
          }

          // Step 4: Should include expected pipeline files
          const fileNames = result.files.map((f) => f.file);
          expect(fileNames).toContain("config.json");
          expect(fileNames).toContain("tokenizer.json");
          expect(fileNames.some((f) => f.startsWith("onnx/") && f.endsWith(".onnx"))).toBe(true);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "should throw for unsupported pipeline task",
        async () => {
          await expect(ModelRegistry.clear_pipeline_cache("invalid-nonexistent-task", LLAMA_MODEL_ID, DEFAULT_MODEL_OPTIONS)).rejects.toThrow("Unsupported pipeline task");
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    describe("error handling", () => {
      it(
        "should throw for empty modelId in is_cached",
        async () => {
          await expect(ModelRegistry.is_cached("")).rejects.toThrow("modelId is required");
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "should throw for empty modelId in clear_cache",
        async () => {
          await expect(ModelRegistry.clear_cache("")).rejects.toThrow("modelId is required");
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "should throw for empty task in is_pipeline_cached",
        async () => {
          await expect(ModelRegistry.is_pipeline_cached("", BERT_MODEL_ID)).rejects.toThrow("task is required");
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "should throw for empty task in clear_pipeline_cache",
        async () => {
          await expect(ModelRegistry.clear_pipeline_cache("", BERT_MODEL_ID)).rejects.toThrow("task is required");
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });
  });
});
