import { AutoModel, PreTrainedModel } from "../../src/transformers.js";
import { AssetLoadingContext } from "../../src/utils/hub/AssetLoadingContext.js";

import { MAX_TEST_EXECUTION_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";
import fs from "node:fs";

// TODO: Set cache folder to a temp directory

describe("Hub", () => {
  describe("Session env", () => {
    it("should use scoped resource path options", () => {
      const context = new AssetLoadingContext({
        revision: "refs/pr/1",
        env: {
          localModelPath: "/scoped-models/",
          remoteHost: "https://models.example.com/",
          remotePathTemplate: "{model}/at/{revision}/",
        },
      });
      const paths = context.resolve("org/model", "config.json");

      expect(paths.localPath).toBe("/scoped-models/org/model/config.json");
      expect(paths.remoteURL).toBe("https://models.example.com/org/model/at/refs%2Fpr%2F1/config.json");
    });

    it("should use scoped Hugging Face token for request headers", () => {
      const context = new AssetLoadingContext({ env: { hfToken: "scoped-token" } });
      const headers = context.getHeaders("https://huggingface.co/org/model/resolve/main/config.json");

      expect(headers.get("Authorization")).toBe("Bearer scoped-token");
    });

    it("should authenticate only the configured custom Hub origin", () => {
      const context = new AssetLoadingContext({
        env: { hfToken: "scoped-token", remoteHost: "https://private-hub.example/api/" },
      });

      const hubHeaders = context.getHeaders("https://private-hub.example/api/org/model/config.json");
      const externalHeaders = context.getHeaders("https://cdn.example/org/model/config.json");
      const downgradedHeaders = context.getHeaders("http://private-hub.example/api/org/model/config.json");
      const otherPortHeaders = context.getHeaders("https://private-hub.example:8443/api/org/model/config.json");

      expect(hubHeaders.get("Authorization")).toBe("Bearer scoped-token");
      expect(externalHeaders.has("Authorization")).toBe(false);
      expect(downgradedHeaders.has("Authorization")).toBe(false);
      expect(otherPortHeaders.has("Authorization")).toBe(false);
    });

    it("should split memoization keys by token", () => {
      const key = (hfToken) => new AssetLoadingContext({ env: { hfToken } }).key("org/model");

      expect(key("hf_token")).toBe(key("hf_token"));
      expect(key("hf_token")).not.toBe(key("hf_other-token"));
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
