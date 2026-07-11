import { jest } from "@jest/globals";

const createInterpreter = jest.fn(() => ({
  computeMask: jest.fn(() => ({ stop: true })),
  commitToken: jest.fn(),
}));
const loadBundledLLGuidance = jest.fn(async () => ({ createInterpreter }));
const loadWasmBinary = jest.fn();
const loadWasmFactory = jest.fn();
const logger = {
  debug: jest.fn(),
  warn: jest.fn(),
};

class LogitsProcessor {}
class LogitsProcessorList extends Array {}
class StoppingCriteria {}

jest.unstable_mockModule("@huggingface/transformers", () => ({
  LogitsProcessor,
  LogitsProcessorList,
  StoppingCriteria,
  env: { useWasmCache: true },
  loadWasmBinary,
  loadWasmFactory,
  logger,
}));

jest.unstable_mockModule("llguidance", () => ({
  loadBundledLLGuidance,
}));

const originalProcess = globalThis.process;
const originalLocationDescriptor = Object.getOwnPropertyDescriptor(globalThis, "location");
const { LlguidanceConstraint } = await import("../dist/index.js");

describe("llguidance WASM loading", () => {
  beforeEach(() => {
    delete globalThis.process;
    Object.defineProperty(globalThis, "location", {
      configurable: true,
      value: { href: "https://example.test/app/" },
    });

    createInterpreter.mockClear();
    loadBundledLLGuidance.mockClear();
    loadWasmBinary.mockReset();
    loadWasmFactory.mockReset();
    logger.debug.mockClear();
    logger.warn.mockClear();
  });

  afterEach(() => {
    globalThis.process = originalProcess;
    if (originalLocationDescriptor) {
      Object.defineProperty(globalThis, "location", originalLocationDescriptor);
    } else {
      delete globalThis.location;
    }
  });

  // Exercises browser-like cache preloading with the package's default CDN asset URLs.
  it("preloads default WASM assets when cache is enabled outside Node", async () => {
    const wasm = new Uint8Array([1, 2, 3]);
    loadWasmBinary.mockResolvedValue(wasm);
    loadWasmFactory.mockResolvedValue("blob:factory-url");

    await LlguidanceConstraint.fromResponseFormat({}, { type: "json_object" });

    expect(loadWasmBinary).toHaveBeenCalledWith("https://cdn.jsdelivr.net/npm/llguidance@0.1.7/wasm/llguidance_wasm_bg.wasm");
    expect(loadWasmFactory).toHaveBeenCalledWith("https://cdn.jsdelivr.net/npm/llguidance@0.1.7/wasm/llguidance_wasm.js");
    expect(loadBundledLLGuidance).toHaveBeenCalledWith({
      wasm,
      wasmFactoryUrl: "blob:factory-url",
    });
  });

  // Verifies custom relative and absolute URLs are normalized before using the transformers cache helpers.
  it("resolves custom cacheable WASM URLs", async () => {
    const wasm = new Uint8Array([4, 5, 6]);
    loadWasmBinary.mockResolvedValue(wasm);
    loadWasmFactory.mockResolvedValue("blob:custom-factory-url");

    await LlguidanceConstraint.fromResponseFormat(
      {},
      { type: "json_object" },
      {
        wasmUrl: "assets/custom.wasm",
        wasmFactoryUrl: new URL("https://cdn.test/custom-factory.js"),
      },
    );

    expect(loadWasmBinary).toHaveBeenCalledWith("https://example.test/app/assets/custom.wasm");
    expect(loadWasmFactory).toHaveBeenCalledWith("https://cdn.test/custom-factory.js");
    expect(loadBundledLLGuidance).toHaveBeenCalledWith({
      wasm,
      wasmFactoryUrl: "blob:custom-factory-url",
    });
  });

  // Ensures a failed preload does not prevent llguidance from loading with the original options.
  it("falls back to uncached options when WASM preload fails", async () => {
    const error = new Error("network failure");
    loadWasmBinary.mockRejectedValue(error);
    loadWasmFactory.mockResolvedValue("blob:factory-url");

    await LlguidanceConstraint.fromResponseFormat({}, { type: "json_object" }, { wasmUrl: "custom.wasm" });

    expect(logger.warn).toHaveBeenCalledWith("Failed to pre-load llguidance WASM binary:", error);
    expect(loadBundledLLGuidance).toHaveBeenCalledWith({ wasmUrl: "custom.wasm" });
  });

  // Avoids cache preloading when the caller already supplied an initialized factory.
  it("skips preloading when a WASM factory is provided", async () => {
    const wasmFactory = jest.fn();

    await LlguidanceConstraint.fromResponseFormat({}, { type: "json_object" }, { wasmFactory });

    expect(loadWasmBinary).not.toHaveBeenCalled();
    expect(loadWasmFactory).not.toHaveBeenCalled();
    expect(loadBundledLLGuidance).toHaveBeenCalledWith({ wasmFactory });
  });
});
