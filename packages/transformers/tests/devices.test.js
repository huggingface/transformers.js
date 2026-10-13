import { getAvailableDevices, env } from "../src/transformers.js";
import { DEVICE_TYPES } from "../src/utils/devices.js";

describe("getAvailableDevices", () => {
  it("returns a non-empty array of supported devices", () => {
    const devices = getAvailableDevices();
    expect(Array.isArray(devices)).toBe(true);
    expect(devices.length).toBeGreaterThan(0);
  });

  it("contains a CPU-capable device (wasm in web environments, cpu in Node.js)", () => {
    const devices = getAvailableDevices();
    // In Node.js the fallback is 'cpu'; in web builds it is 'wasm'. Tests run in Node.js.
    if (env.backends?.onnx) {
      // Node.js runtime: onnxruntime-node present.
      expect(devices).toContain("cpu");
    } else {
      expect(devices).toContain("wasm");
    }
  });

  it("only reports known device types", () => {
    const known = new Set(Object.keys(DEVICE_TYPES));
    for (const device of getAvailableDevices()) {
      expect(known.has(device)).toBe(true);
    }
  });

  it("returns a fresh copy each call (mutating the result does not affect later calls)", () => {
    const first = getAvailableDevices();
    first.push("not-a-device");
    const second = getAvailableDevices();
    expect(second).not.toContain("not-a-device");
  });
});
