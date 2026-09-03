// Helper functions used when initialising the testing environment.

import { initOnnxTestBackend } from "@huggingface/transformers-onnxruntime/testing";

/**
 * A workaround to define a new backend for onnxruntime, which
 * will not throw an error when running tests with jest.
 * For more information, see: https://github.com/jestjs/jest/issues/11864#issuecomment-1261468011
 */
export function init() {
  initOnnxTestBackend();
}

export const MAX_TOKENIZER_LOAD_TIME = 32_000; // 32 seconds
export const MAX_FEATURE_EXTRACTOR_LOAD_TIME = 10_000; // 10 seconds
export const MAX_PROCESSOR_LOAD_TIME = 10_000; // 10 seconds
export const MAX_MODEL_LOAD_TIME = 15_000; // 15 seconds
export const MAX_TEST_EXECUTION_TIME = 60_000; // 60 seconds
export const MAX_MODEL_DISPOSE_TIME = 1_000; // 1 second

export const MAX_TEST_TIME = MAX_MODEL_LOAD_TIME + MAX_TEST_EXECUTION_TIME + MAX_MODEL_DISPOSE_TIME;

export const DEFAULT_MODEL_OPTIONS = Object.freeze({
  dtype: "fp32",
});

expect.extend({
  toBeCloseToNested(received, expected, numDigits = 2) {
    const compare = (received, expected, path = "") => {
      if (typeof received === "number" && typeof expected === "number" && (!Number.isInteger(received) || !Number.isInteger(expected))) {
        const pass = Math.abs(received - expected) < Math.pow(10, -numDigits);
        return {
          pass,
          message: () => (pass ? `✓ At path '${path}': expected ${received} not to be close to ${expected} with tolerance of ${numDigits} decimal places` : `✗ At path '${path}': expected ${received} to be close to ${expected} with tolerance of ${numDigits} decimal places`),
        };
      } else if (Array.isArray(received) && Array.isArray(expected)) {
        if (received.length !== expected.length) {
          return {
            pass: false,
            message: () => `✗ At path '${path}': array lengths differ. Received length ${received.length}, expected length ${expected.length}`,
          };
        }
        for (let i = 0; i < received.length; i++) {
          const result = compare(received[i], expected[i], `${path}[${i}]`);
          if (!result.pass) return result;
        }
      } else if (typeof received === "object" && typeof expected === "object" && received !== null && expected !== null) {
        const receivedKeys = Object.keys(received);
        const expectedKeys = Object.keys(expected);
        if (receivedKeys.length !== expectedKeys.length) {
          return {
            pass: false,
            message: () => `✗ At path '${path}': object keys length differ. Received keys: ${JSON.stringify(receivedKeys)}, expected keys: ${JSON.stringify(expectedKeys)}`,
          };
        }
        for (const key of receivedKeys) {
          if (!expected.hasOwnProperty(key)) {
            return {
              pass: false,
              message: () => `✗ At path '${path}': key '${key}' found in received but not in expected`,
            };
          }
          const result = compare(received[key], expected[key], `${path}.${key}`);
          if (!result.pass) return result;
        }
      } else {
        const pass = received === expected;
        return {
          pass,
          message: () => (pass ? `✓ At path '${path}': expected ${JSON.stringify(received)} not to equal ${JSON.stringify(expected)}` : `✗ At path '${path}': expected ${JSON.stringify(received)} to equal ${JSON.stringify(expected)}`),
        };
      }
      return { pass: true };
    };

    return compare(received, expected);
  },
});
