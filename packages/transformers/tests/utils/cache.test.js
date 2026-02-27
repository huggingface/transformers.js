import { ModelRegistry } from "../../src/transformers.js";

import { MAX_TEST_EXECUTION_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";

const LLAMA_MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM";
const BERT_MODEL_ID = "hf-internal-testing/tiny-random-BertModel";

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

    describe.skip("clear_cache", () => {
      it(
        "should return clear result with correct shape",
        async () => {
          const result = await ModelRegistry.clear_cache(BERT_MODEL_ID, DEFAULT_MODEL_OPTIONS);
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
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    describe.skip("clear_pipeline_cache", () => {
      it(
        "should return clear result for text-generation pipeline",
        async () => {
          const result = await ModelRegistry.clear_pipeline_cache("text-generation", LLAMA_MODEL_ID, DEFAULT_MODEL_OPTIONS);
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
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });
  });
});
