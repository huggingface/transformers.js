import { env, LlamaForCausalLM, AutoTokenizer } from "../../src/transformers.js";
import { init, MAX_TEST_EXECUTION_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";

// Initialise the testing environment
init();

/**
 * A naive custom cache implementation that fetches files directly from the
 * Hugging Face Hub and stores them in an internal (in-memory) map.
 * This satisfies the CacheInterface contract (`match` + `put`).
 */
class NaiveFetchCache {
  constructor() {
    /** @type {Map<string, Response>} */
    this.cache = new Map();
  }

  async match(request) {
    const cached = this.cache.get(request);
    if (cached) {
      return cached.clone();
    }

    // Not in cache — attempt a fresh fetch from the URL.
    try {
      const response = await fetch(request);
      if (response.ok) {
        this.cache.set(request, response);
        return response.clone();
      }
    } catch {
      // Ignore fetch errors (e.g., invalid URLs like local paths) — treat as cache miss
    }
    return undefined;
  }

  async put(request, response) {
    if (!this.cache.has(request)) {
      this.cache.set(request, response);
    }
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
