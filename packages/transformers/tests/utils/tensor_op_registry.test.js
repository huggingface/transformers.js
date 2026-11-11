import { jest } from "@jest/globals";

// Regression tests for https://github.com/huggingface/transformers.js/issues/1806:
// auxiliary operator sessions (TensorOpRegistry) must not fall back to ORT's
// default provider list in web environments (which may probe a GPU adapter even
// when the main model is pinned to WASM), and must be overridable by consumers.

jest.unstable_mockModule("../../src/backends/onnx.js", () => ({
  createInferenceSession: jest.fn(async () => ({ run: async (feed) => feed })),
  runInferenceSession: jest.fn(async (session, ortFeed) => ortFeed),
  isONNXProxy: jest.fn(() => false),
  deviceToExecutionProviders: jest.fn(() => ["wasm"]),
  // minimal stub: utils/tensor.js only needs these bindings at import time,
  // and these tests never construct real tensors
  Tensor: class {},
  isONNXTensor: (x) => x instanceof Error,
}));

describe("TensorOpRegistry.session_options", () => {
  it("stays empty in non-web environments (native defaults unchanged)", async () => {
    const { TensorOpRegistry } = await import("../../src/ops/registry.js");
    expect(TensorOpRegistry.session_options).toEqual({});
  });

  it("pins auxiliary sessions to WASM in web environments", async () => {
    // session_options is initialized at class-evaluation time from the
    // environment, so a web-env instance needs a fresh module registry.
    jest.resetModules();
    jest.unstable_mockModule("../../src/env.js", () => ({
      apis: Object.freeze({ IS_WEB_ENV: true, IS_BROWSER_ENV: true }),
    }));
    const { TensorOpRegistry } = await import("../../src/ops/registry.js");
    expect(TensorOpRegistry.session_options).toEqual({ executionProviders: ["wasm"] });

    const { createInferenceSession } = await import("../../src/backends/onnx.js");
    await TensorOpRegistry.matmul;
    expect(createInferenceSession).toHaveBeenCalled();
    expect(createInferenceSession.mock.calls[0][1]).toEqual({ executionProviders: ["wasm"] });
  }, 2000);

  it("is overridable by consumers", async () => {
    const { TensorOpRegistry } = await import("../../src/ops/registry.js");
    const original = TensorOpRegistry.session_options;
    try {
      TensorOpRegistry.session_options = { executionProviders: ["webgpu"] };
      await TensorOpRegistry.slice;
      const { createInferenceSession } = await import("../../src/backends/onnx.js");
      expect(createInferenceSession).toHaveBeenCalled();
      expect(createInferenceSession.mock.calls.at(-1)[1]).toEqual({ executionProviders: ["webgpu"] });
    } finally {
      TensorOpRegistry.session_options = original;
    }
  }, 2000);
});
