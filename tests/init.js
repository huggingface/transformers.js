// Helper functions used when initialising the testing environment.

// Import Node typing utilities
import * as types from "node:util/types";

// Import onnxruntime-node's default backend
import { onnxruntimeBackend } from "onnxruntime-node/dist/backend";
import * as ONNX_COMMON from "onnxruntime-common";

/**
 * A workaround to define a new backend for onnxruntime, which
 * will not throw an error when running tests with jest.
 * For more information, see: https://github.com/jestjs/jest/issues/11864#issuecomment-1261468011
 */
export function init() {
  // In rare cases (specifically when running unit tests with GitHub actions), possibly due to
  // a large number of concurrent executions, onnxruntime might fallback to use the WASM backend.
  // In this case, we set the number of threads to 1 to avoid errors like:
  //  - `TypeError: The worker script or module filename must be an absolute path or a relative path starting with './' or '../'. Received "blob:nodedata:..."`
  ONNX_COMMON.env.wasm.numThreads = 1;

  let registerBackend = ONNX_COMMON.registerBackend;

  // Define the constructors to monkey-patch
  const TYPED_ARRAYS_CONSTRUCTOR_NAMES = ["Int8Array", "Int16Array", "Int32Array", "BigInt64Array", "Uint8Array", "Uint8ClampedArray", "Uint16Array", "Uint32Array", "BigUint64Array", "Float16Array", "Float32Array", "Float64Array"];

  // Keep a reference to the original initialization method
  const originalMethod = onnxruntimeBackend.init;

  // Monkey-patch the initialization function
  onnxruntimeBackend.init = function (...args) {
    // There is probably a better way to do this
    Array.isArray = (x) => typeof x === "object" && x !== null && typeof x.length === "number" && x?.constructor.toString() === Array.toString();

    // For each typed array constructor
    for (const ctorName of TYPED_ARRAYS_CONSTRUCTOR_NAMES) {
      // Get the constructor from the current context
      const ctor = globalThis[ctorName];
      if (ctor === undefined) continue; // If unavailable, skip the patching

      // Get the corresponding test function from the `util` module
      const value = types[`is${ctorName}`].bind(types);

      // Monkey-patch the constructor so "x instanceof ctor" returns "types[`is${ctorName}`](x)"
      Object.defineProperty(ctor, Symbol.hasInstance, {
        value,
        writable: true, // writable=true is necessary to overwrite the default implementation (and allow subsequent overwrites)
        configurable: false,
        enumerable: false,
      });
    }

    // Call the original method
    return originalMethod.apply(this, args);
  };

  // Register the backend with the highest priority, so it is used instead of the default one
  registerBackend("test", onnxruntimeBackend, Number.POSITIVE_INFINITY);
}

export const MAX_TOKENIZER_LOAD_TIME = 10_000; // 10 seconds
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
      if (typeof received === "number" && typeof expected === "number" && !Number.isInteger(received) && !Number.isInteger(expected)) {
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
