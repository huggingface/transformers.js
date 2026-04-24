// Generate per-module API markdown from the library's JSDoc comments.
// We only document items that are available in the public API.

import fs from "node:fs";
import path from "node:path";
import url from "node:url";

import { loadProject } from "./lib/load.mjs";
import { renderModule } from "./lib/render-api.mjs";

const docs = path.dirname(path.dirname(url.fileURLToPath(import.meta.url)));
const root = path.dirname(docs);
const outputDir = path.join(root, "docs", "source", "api");

const { ir, publicNames } = loadProject(root);

clearExistingMarkdown();

for (const mod of ir.modules) {
  if (!hasPublicContent(mod, publicNames)) {
    console.log(`skipped ${mod.name}.md — no public content`);
    continue;
  }
  const output = renderModule(mod, ir, { publicNames });
  const outputPath = path.resolve(outputDir, `${mod.name}.md`);
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, output);
  console.log(`wrote ${mod.name}.md`);
}

function hasPublicContent(mod, publicNames) {
  const isPublic = (item) => publicNames.has(item.name);
  return mod.classes.some(isPublic) || mod.functions.some(isPublic) || mod.constants.some(isPublic) || mod.typedefs.length > 0 || mod.callbacks.length > 0;
}

function clearExistingMarkdown() {
  if (!fs.existsSync(outputDir)) return;
  for (const entry of fs.readdirSync(outputDir, { recursive: true })) {
    if (entry.endsWith(".md")) fs.unlinkSync(path.join(outputDir, entry));
  }
}
