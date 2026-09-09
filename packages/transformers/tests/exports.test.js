import { readdirSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import * as transformers from "../src/transformers.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const MODELS_DIR = join(__dirname, "..", "src", "models");

function findModelingFiles(dir) {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);

    if (entry.isDirectory()) {
      return findModelingFiles(path);
    }
    return entry.isFile() && entry.name.startsWith("modeling_") && entry.name.endsWith(".js") ? [path] : [];
  });
}

function isPublicModelingFile(file) {
  return !file.endsWith(join("models", "modeling_utils.js"));
}

describe("Public exports", () => {
  it("exports every public modeling_* symbol from the root entry point", async () => {
    const missing = [];

    for (const file of findModelingFiles(MODELS_DIR).filter(isPublicModelingFile).sort()) {
      const moduleExports = await import(pathToFileURL(file).href);

      for (const exportName of Object.keys(moduleExports)) {
        if (!Object.hasOwn(transformers, exportName)) {
          missing.push(`${relative(MODELS_DIR, file)}: ${exportName}`);
        }
      }
    }

    expect(missing).toEqual([]);
  });
});

describe("supportedDevices public export", () => {
  it("getSupportedDevices returns a frozen snapshot", () => {
    const devices = transformers.env.backends.onnx.getSupportedDevices();
    expect(Object.isFrozen(devices)).toBe(true);

    const before = devices.slice();

    // Attempt to mutate the public snapshot. This throws in strict mode (ESM) and
    // is silently ignored otherwise; either way the snapshot must stay intact.
    try {
      devices.length = 0;
      devices.push("malicious-device");
      devices[0] = "malicious-device";
    } catch {
      // Expected in strict mode (frozen object).
    }

    expect(devices).toEqual(before);
    expect(devices.length).toBe(before.length);
  });

  it("mutating getSupportedDevices() result does not affect internal state", () => {
    const first = transformers.env.backends.onnx.getSupportedDevices();
    expect(first.length).toBeGreaterThan(0);
    const someDevice = first[0];

    // Attempt to mutate the returned array.
    try {
      first.length = 0;
      first.push("malicious-device");
    } catch {
      // ignored
    }

    // Re-fetch and confirm unaffected
    const second = transformers.env.backends.onnx.getSupportedDevices();
    expect(second.length).toBeGreaterThan(0);
    expect(() => transformers.env.backends.onnx.deviceToExecutionProviders(someDevice)).not.toThrow();
  });

  it("deviceToExecutionProviders returns a defensive copy", () => {
    const result = transformers.env.backends.onnx.deviceToExecutionProviders("auto");
    expect(result.length).toBeGreaterThan(0);
    const someDevice = result[0];

    // Mutate the returned array
    try {
      result.length = 0;
      result.push("malicious-device");
    } catch {
      // ignored
    }

    // Subsequent call returns fresh, unaffected result
    const result2 = transformers.env.backends.onnx.deviceToExecutionProviders("auto");
    expect(result2.length).toBeGreaterThan(0);
    expect(() => transformers.env.backends.onnx.deviceToExecutionProviders(someDevice)).not.toThrow();
  });

  it("default branch returns a copy and does not expose internal array", () => {
    const defaultBefore = transformers.env.backends.onnx.deviceToExecutionProviders();
    const defaultAlso = transformers.env.backends.onnx.deviceToExecutionProviders(null);
    expect(defaultAlso).toEqual(defaultBefore);

    // Mutation of returned array does not affect internal state
    const returned = transformers.env.backends.onnx.deviceToExecutionProviders();
    try {
      returned.length = 0;
    } catch {
      // ignored
    }
    expect(transformers.env.backends.onnx.deviceToExecutionProviders()).toEqual(transformers.env.backends.onnx.deviceToExecutionProviders(null));
  });

  it("gpu branch returns a fresh array and does not expose internal array", () => {
    const gpuDevices = transformers.env.backends.onnx.deviceToExecutionProviders("gpu");
    expect(Array.isArray(gpuDevices)).toBe(true);
    // Each call returns a fresh array (filter creates new array)
    const gpuAgain = transformers.env.backends.onnx.deviceToExecutionProviders("gpu");
    expect(gpuAgain).toEqual(gpuDevices);
    expect(gpuAgain).not.toBe(gpuDevices); // different reference
  });

  it("specific device returns execution providers array", () => {
    const devices = transformers.env.backends.onnx.getSupportedDevices();
    expect(devices.length).toBeGreaterThan(0);
    const someDevice = devices[0];

    const eps = transformers.env.backends.onnx.deviceToExecutionProviders(someDevice);
    expect(Array.isArray(eps)).toBe(true);
    expect(eps.length).toBeGreaterThan(0);

    // Returns fresh array each call
    const epsAgain = transformers.env.backends.onnx.deviceToExecutionProviders(someDevice);
    expect(epsAgain).toEqual(eps);
    expect(epsAgain).not.toBe(eps);
  });

  it("unsupported device throws descriptive error", () => {
    expect(() => transformers.env.backends.onnx.deviceToExecutionProviders("unsupported-device-xyz")).toThrow("Unsupported device");
  });

  it("defaultDevices mutation does not affect internal state", () => {
    const defaultBefore = transformers.env.backends.onnx.deviceToExecutionProviders();
    const returned = transformers.env.backends.onnx.deviceToExecutionProviders();
    try {
      returned.push("malicious-device");
    } catch {
      // ignored
    }
    expect(transformers.env.backends.onnx.deviceToExecutionProviders()).toEqual(defaultBefore);
    expect(transformers.env.backends.onnx.deviceToExecutionProviders()).not.toContain("malicious-device");
  });
});
