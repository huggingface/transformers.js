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
  it("is frozen and cannot be mutated by consumers", () => {
    expect(Object.isFrozen(transformers.supportedDevices)).toBe(true);

    const before = transformers.supportedDevices.slice();

    // Attempt to mutate the public snapshot. This throws in strict mode (ESM) and
    // is silently ignored otherwise; either way the snapshot must stay intact.
    try {
      transformers.supportedDevices.length = 0;
      transformers.supportedDevices.push("malicious-device");
      transformers.supportedDevices[0] = "malicious-device";
    } catch {
      // Expected in strict mode (frozen object).
    }

    expect(transformers.supportedDevices).toEqual(before);
    expect(transformers.supportedDevices.length).toBe(before.length);
  });

  it("consumer mutation cannot affect deviceToExecutionProviders", () => {
    const autoBefore = transformers.deviceToExecutionProviders("auto");
    const snapshot = transformers.supportedDevices;
    expect(snapshot.length).toBeGreaterThan(0);
    const someDevice = snapshot[0];

    // Attempt to corrupt the public export.
    try {
      transformers.supportedDevices.length = 0;
    } catch {
      // ignored
    }

    // The internal array is private, so provider selection is unaffected.
    expect(transformers.deviceToExecutionProviders("auto")).toEqual(autoBefore);
    expect(() => transformers.deviceToExecutionProviders(someDevice)).not.toThrow();
  });

  it("mutation of the array returned by deviceToExecutionProviders('auto') cannot affect later calls", () => {
    const autoExpected = transformers.deviceToExecutionProviders("auto").slice();
    expect(autoExpected.length).toBeGreaterThan(0);
    const someDevice = transformers.supportedDevices[0];

    // Mutate the *returned* array (the second public path flagged in review).
    const autoReturned = transformers.deviceToExecutionProviders("auto");
    try {
      autoReturned.length = 0;
      autoReturned.push("malicious-device");
    } catch {
      // ignored (frozen/immutable in some environments)
    }

    // Internal state is private, so subsequent calls return the full, unmutated list.
    expect(transformers.deviceToExecutionProviders("auto")).toEqual(autoExpected);
    expect(() => transformers.deviceToExecutionProviders(someDevice)).not.toThrow();
  });
});
