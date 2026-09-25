// Reads every `.js` file under `src/` and returns the IR, the public-export set and the task
// catalog. Both the API-markdown and skill renderers consume it.

import fs from "node:fs";
import path from "node:path";

import { extractEntities } from "./structure.mjs";
import { listFiles } from "./fs.mjs";
import { buildIR } from "./ir.mjs";
import { collectPublicExports } from "./exports.mjs";
import { extractTaskCatalog } from "./tasks.mjs";

export function loadProject(root) {
  const srcDir = path.join(root, "src");
  const fileEntities = listFiles(srcDir, ".js").map((file) => ({
    file,
    entities: extractEntities(fs.readFileSync(file, "utf8"), file),
  }));
  return {
    ir: buildIR(fileEntities),
    publicNames: collectPublicExports(path.join(srcDir, "transformers.js")),
    tasks: extractTaskCatalog(path.join(srcDir, "pipelines", "index.js")),
  };
}
