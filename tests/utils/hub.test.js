import { AutoModel, PreTrainedModel } from "../../src/models.js";

import { MAX_TEST_EXECUTION_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";
import fs from "fs";

// TODO: Set cache folder to a temp directory

describe("Hub", () => {
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
