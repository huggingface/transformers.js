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
