import { applyExecutionProviderOptions } from "../../src/models/session.js";

describe("applyExecutionProviderOptions (transformers.js_config.execution_provider_options)", () => {
  it("leaves the providers unchanged without options", () => {
    expect(applyExecutionProviderOptions(["webgpu"], undefined)).toEqual(["webgpu"]);
    expect(applyExecutionProviderOptions(["webgpu"], {})).toEqual(["webgpu"]);
    expect(applyExecutionProviderOptions(undefined, { a: 1 })).toBeUndefined();
  });

  it("turns string providers into objects carrying the options", () => {
    const names = ["/lstm/LSTM", "/shared/LSTM"];
    expect(applyExecutionProviderOptions(["webgpu"], { forceCpuNodeNames: names })).toEqual([{ name: "webgpu", forceCpuNodeNames: names }]);
  });

  it("applies to every provider in the list and keeps the provider's own fields", () => {
    const out = applyExecutionProviderOptions([{ name: "webnn", deviceType: "gpu" }, "wasm"], { deviceType: "cpu", powerPreference: "high-performance" });
    expect(out).toEqual([
      { name: "webnn", deviceType: "gpu", powerPreference: "high-performance" },
      { name: "wasm", deviceType: "cpu", powerPreference: "high-performance" },
    ]);
  });

  it("never lets the options rename a provider", () => {
    expect(applyExecutionProviderOptions(["webgpu"], { name: "wasm" })).toEqual([{ name: "webgpu" }]);
  });
});
