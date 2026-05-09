// Shared loader: reads every `.js` file under `src/`, extracts TS-backed
// entities, builds the IR, and returns it along with the public-export set
// and the task catalog. Both the API-markdown and skill renderers consume it.

import fs from "node:fs";
import path from "node:path";

import { extractEntities } from "./structure.mjs";
import { buildIR } from "./ir.mjs";
import { collectPublicExports } from "./exports.mjs";
import { extractTaskCatalog } from "./tasks.mjs";

export function loadProject(root) {
  const srcDir = path.join(root, "src");
  const fileEntities = collectJsFiles(srcDir).map((file) => ({
    file,
    entities: extractEntities(fs.readFileSync(file, "utf8"), file),
  }));
  return {
    ir: buildIR(fileEntities),
    publicNames: collectPublicExports(path.join(srcDir, "transformers.js")),
    tasks: extractTaskCatalog(path.join(srcDir, "pipelines", "index.js")),
  };
}

function collectJsFiles(dir) {
  const out = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...collectJsFiles(full));
    else if (entry.name.endsWith(".js")) out.push(full);
  }
  return out;
}
