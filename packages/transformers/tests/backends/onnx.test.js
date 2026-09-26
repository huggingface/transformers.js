import { jest } from "@jest/globals";

// onnx.js takes the ONNX runtime from this global first, ahead of any environment
// branch, so a fake here keeps the test free of the native onnxruntime binaries.
const ORT_SYMBOL = Symbol.for("onnxruntime");
const create = jest.fn();
globalThis[ORT_SYMBOL] = {
  env: { wasm: {} },
  InferenceSession: { create },
  Tensor: class {},
};

// `apis` is frozen, so the web path has to be selected by mocking the module.
jest.unstable_mockModule("../../src/env.js", () => ({
  env: { backends: { onnx: {} }, useWasmCache: false, logLevel: undefined },
  apis: {
    IS_NODE_ENV: false,
    IS_WEB_ENV: true,
    IS_WEBNN_AVAILABLE: false,
    IS_WEBGPU_AVAILABLE: false,
    IS_DENO_WEB_RUNTIME: false,
    IS_SAFARI_BELOW_26: false,
  },
  LogLevel: { WARNING: 2 },
}));

const { createInferenceSession, runInferenceSession } = await import("../../src/backends/onnx.js");

describe("web-path serialization chains", () => {
  it("a failed session creation does not fail the sessions created after it", async () => {
    create.mockRejectedValueOnce(new Error("cuda provider missing"));
    create.mockResolvedValue({ run: async () => ({}) });

    await expect(createInferenceSession("first", {}, {})).rejects.toThrow("cuda provider missing");

    // Before the fix the chain itself was left rejected, so `.then(load)` skipped
    // `load` and re-threw the first error here without ever calling `create`.
    const session = await createInferenceSession("second", {}, {});
    expect(session).toBeDefined();
    expect(create).toHaveBeenCalledTimes(2);
  });

  it("keeps creations serialized in call order", async () => {
    const order = [];
    create.mockImplementation(async (name) => {
      order.push(`start:${name}`);
      await new Promise((r) => setTimeout(r, name === "slow" ? 20 : 0));
      order.push(`end:${name}`);
      return { run: async () => ({}) };
    });

    await Promise.all([createInferenceSession("slow", {}, {}), createInferenceSession("fast", {}, {})]);
    expect(order).toEqual(["start:slow", "end:slow", "start:fast", "end:fast"]);
  });

  it("a failed inference run does not fail the runs after it", async () => {
    const run = jest.fn().mockRejectedValueOnce(new Error("session already started")).mockResolvedValue({ out: 1 });
    const session = { run };

    await expect(runInferenceSession(session, {})).rejects.toThrow("session already started");
    await expect(runInferenceSession(session, {})).resolves.toEqual({ out: 1 });
    expect(run).toHaveBeenCalledTimes(2);
  });
});
