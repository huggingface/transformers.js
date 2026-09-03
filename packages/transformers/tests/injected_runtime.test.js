import { describe, expect, it, jest } from "@jest/globals";

const ORT_SYMBOL = Symbol.for("onnxruntime");

/**
 * Evaluate `backends/onnx.js` afresh with `globalThis[Symbol.for('onnxruntime')]` set to `runtime`, and return
 * the devices it resolved. The module reads the symbol once, when it is evaluated, so every case needs its own
 * module registry.
 */
async function devicesWith(runtime) {
  let devices;
  globalThis[ORT_SYMBOL] = runtime;
  try {
    await jest.isolateModulesAsync(async () => {
      const { deviceToExecutionProviders } = await import("../src/backends/onnx.js");
      devices = { auto: deviceToExecutionProviders("auto"), defaults: deviceToExecutionProviders(null) };
    });
  } finally {
    delete globalThis[ORT_SYMBOL];
  }
  return devices;
}

/** The shape an injected runtime has: the onnxruntime API surface, with an `env` that may carry a version stamp. */
const runtime = (versions) => ({
  env: versions ? { versions: { common: "0.0.0", ...versions } } : {},
  Tensor: class {},
  InferenceSession: { create() {} },
});

describe("Injected ONNX runtime (globalThis[Symbol.for('onnxruntime')])", () => {
  it("an injected onnxruntime-web gets the web device list, whatever the process", async () => {
    // This test runs under Node, where the environment alone would pick the node list.
    const { auto, defaults } = await devicesWith(runtime({ web: "0.0.0" }));
    expect(auto).toContain("wasm");
    expect(auto).not.toContain("cpu");
    expect(defaults).toEqual(["wasm"]);
  });

  it("an injected onnxruntime-node gets the node device list", async () => {
    const { auto, defaults } = await devicesWith(runtime({ node: "0.0.0" }));
    expect(auto).toContain("cpu");
    expect(auto).not.toContain("wasm");
    expect(defaults).toEqual(["cpu"]);
  });

  it("a custom runtime, stamped with no version, gets no device list and keeps its own defaults", async () => {
    const { auto, defaults } = await devicesWith(runtime(null));
    expect(auto).toEqual([]);
    expect(defaults).toBeUndefined();
  });
});
