import { env, LlamaForCausalLM, AutoTokenizer } from "../../src/transformers.js";
import { init, MAX_TEST_EXECUTION_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";

// Initialise the testing environment
init();

/**
 * A naive custom cache implementation that fetches files directly from the
 * Hugging Face Hub on every `match` call (i.e., no real caching).
 * This satisfies the CacheInterface contract (`match` + `put`).
 */
class NaiveFetchCache {
  async match(request) {
    // `request` is a URL string (either the local path or the remote HF Hub URL).
    // We attempt to fetch it and return the Response directly.
    try {
      const response = await fetch(request);
      if (response.ok) {
        return response;
      }
    } catch {
      // Ignore fetch errors (e.g., invalid URLs like local paths) — treat as cache miss
    }
    return undefined;
  }

  async put(request, response) {
    // No-op: we don't actually store anything.
  }
}

describe("Custom cache", () => {
  // Store original env values so we can restore them after tests
  const originalUseCustomCache = env.useCustomCache;
  const originalCustomCache = env.customCache;
  const originalUseBrowserCache = env.useBrowserCache;
  const originalUseFSCache = env.useFSCache;
  const originalAllowLocalModels = env.allowLocalModels;

  beforeAll(() => {
    // Disable all other caching mechanisms so only customCache is used
    env.useCustomCache = true;
    env.customCache = new NaiveFetchCache();
    env.useBrowserCache = false;
    env.useFSCache = false;
    env.allowLocalModels = false;
  });

  afterAll(() => {
    // Restore original env values
    env.useCustomCache = originalUseCustomCache;
    env.customCache = originalCustomCache;
    env.useBrowserCache = originalUseBrowserCache;
    env.useFSCache = originalUseFSCache;
    env.allowLocalModels = originalAllowLocalModels;
  });

  it(
    "should load a model using custom cache (standard)",
    async () => {
      const model_id = "onnx-internal-testing/tiny-random-LlamaForCausalLM-ONNX";

      const tokenizer = await AutoTokenizer.from_pretrained(model_id);
      const model = await LlamaForCausalLM.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);

      const inputs = await tokenizer("Hello");
      const output = await model(inputs);

      expect(output.logits).toBeDefined();
      const expected_shape = [...inputs.input_ids.dims, model.config.vocab_size];
      expect(output.logits.dims).toEqual(expected_shape);

      await model.dispose();
    },
    MAX_TEST_EXECUTION_TIME,
  );

  it(
    "should load a model using custom cache (external data)",
    async () => {
      const model_id = "onnx-internal-testing/tiny-random-LlamaForCausalLM-ONNX_external";

      const tokenizer = await AutoTokenizer.from_pretrained(model_id);
      const model = await LlamaForCausalLM.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);

      const inputs = await tokenizer("Hello");
      const output = await model(inputs);

      expect(output.logits).toBeDefined();
      const expected_shape = [...inputs.input_ids.dims, model.config.vocab_size];
      expect(output.logits.dims).toEqual(expected_shape);

      await model.dispose();
    },
    MAX_TEST_EXECUTION_TIME,
  );
});
