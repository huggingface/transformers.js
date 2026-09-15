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

describe("getSupportedDevices public export", () => {
  it("returns a frozen snapshot", () => {
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

  it("each call returns a fresh snapshot", () => {
    const first = transformers.env.backends.onnx.getSupportedDevices();
    const second = transformers.env.backends.onnx.getSupportedDevices();
    expect(first).toEqual(second);
    expect(first).not.toBe(second); // different reference
  });

  it("snapshot is non-empty when devices are available", () => {
    const devices = transformers.env.backends.onnx.getSupportedDevices();
    expect(devices.length).toBeGreaterThan(0);
  });
});

describe("deviceToExecutionProviders public export", () => {
  it("returns a defensive copy", () => {
    const result = transformers.env.backends.onnx.deviceToExecutionProviders("auto");
    expect(result.length).toBeGreaterThan(0);

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
  });

  it("each call returns a fresh array", () => {
    const first = transformers.env.backends.onnx.deviceToExecutionProviders("auto");
    const second = transformers.env.backends.onnx.deviceToExecutionProviders("auto");
    expect(first).toEqual(second);
    expect(first).not.toBe(second); // different reference
  });

  it("specific device returns execution providers array", () => {
    const devices = transformers.env.backends.onnx.getSupportedDevices();
    const someDevice = devices[0];

    const eps = transformers.env.backends.onnx.deviceToExecutionProviders(someDevice);
    expect(Array.isArray(eps)).toBe(true);
    expect(eps.length).toBeGreaterThan(0);
  });

  it("unsupported device throws descriptive error", () => {
    expect(() => transformers.env.backends.onnx.deviceToExecutionProviders("unsupported-device-xyz")).toThrow("Unsupported device");
  });
});
