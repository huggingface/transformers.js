import { AutoModel, PreTrainedModel } from "../../src/transformers.js";
import { FileCache } from "../../src/utils/cache/FileCache.js";
import { buildResourcePaths } from "../../src/utils/hub.js";

import { MAX_TEST_EXECUTION_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";
import fs from "node:fs";

// TODO: Set cache folder to a temp directory

describe("Hub", () => {
  describe("Resource paths", () => {
    it("should version tokenizer metadata browser cache keys", () => {
      const tokenizerConfig = buildResourcePaths("onnx-community/model", "tokenizer_config.json");
      const tokenizer = buildResourcePaths("onnx-community/model", "tokenizer.json");

      expect(tokenizerConfig.proposedCacheKey).toBe(`${tokenizerConfig.remoteURL}?transformersjs_metadata_v=2`);
      expect(tokenizer.proposedCacheKey).toBe(tokenizer.remoteURL);
    });

    it("should version tokenizer metadata filesystem cache keys", () => {
      const cache = new FileCache("cache");
      const tokenizerConfig = buildResourcePaths("onnx-community/model", "tokenizer_config.json", {}, cache);
      const tokenizer = buildResourcePaths("onnx-community/model", "tokenizer.json", {}, cache);

      expect(tokenizerConfig.proposedCacheKey).toBe("transformersjs_metadata_v2/onnx-community/model/tokenizer_config.json");
      expect(tokenizer.proposedCacheKey).toBe("onnx-community/model/tokenizer.json");
    });
  });

  describe("Loading models", () => {
    it(
      "should load a model from the local cache",
      async () => {
        // 1. Local model exists (doesn't matter about status of remote file since local is tried first)
        const model = await AutoModel.from_pretrained("hf-internal-testing/tiny-random-T5ForConditionalGeneration", DEFAULT_MODEL_OPTIONS);
        expect(model).toBeInstanceOf(PreTrainedModel);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "should load a model from the remote cache",
      async () => {
        // 2. Local model doesn't exist, remote file exists
        // This tests that fallback functionality is working
        const model = await AutoModel.from_pretrained("hf-internal-testing/tiny-random-T5ForConditionalGeneration", DEFAULT_MODEL_OPTIONS);
        expect(model).toBeInstanceOf(PreTrainedModel);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "should fail to load a model",
      async () => {
        // 3. Local model doesn't exist, remote file doesn't exist
        // This tests that error handling is working.
        await expect(AutoModel.from_pretrained("hf-internal-testing/this-model-does-not-exist", DEFAULT_MODEL_OPTIONS)).rejects.toBeInstanceOf(Error);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    const localPath = "./models/hf-internal-testing/tiny-random-T5ForConditionalGeneration";
    (fs.existsSync(localPath) ? it : it.skip)(
      "should load a model from a local path",
      async () => {
        // 4. Ensure we can load a model from a local path
        const model = await AutoModel.from_pretrained(localPath, DEFAULT_MODEL_OPTIONS);
        expect(model).toBeInstanceOf(PreTrainedModel);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
});
