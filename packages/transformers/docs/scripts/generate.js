// Generate per-module API markdown from the library's JSDoc comments.
// We only document items that are available in the public API.

import fs from "node:fs";
import path from "node:path";
import url from "node:url";

import { extractEntities } from "./lib/structure.mjs";
import { buildIR } from "./lib/ir.mjs";
import { renderModule } from "./lib/render-api.mjs";
import { collectPublicExports } from "./lib/exports.mjs";

const docs = path.dirname(path.dirname(url.fileURLToPath(import.meta.url)));
const root = path.dirname(docs);
const srcDir = path.join(root, "src");
const outputDir = path.join(root, "docs", "source", "api");

const fileEntities = collectJsFiles(srcDir).map((file) => ({
  file,
  entities: extractEntities(fs.readFileSync(file, "utf8"), file),
}));

const ir = buildIR(fileEntities);
const publicNames = collectPublicExports(path.join(srcDir, "transformers.js"));

clearExistingMarkdown();

for (const mod of ir.modules) {
  const output = renderModule(mod, ir, { publicNames });
  const outputPath = path.resolve(outputDir, `${mod.name}.md`);
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, output);
  const empty = output.trim().split("\n").length <= 2 ? "  (empty — no public content)" : "";
  console.log(`wrote ${mod.name}.md${empty}`);
}

function clearExistingMarkdown() {
  if (!fs.existsSync(outputDir)) return;
  for (const entry of fs.readdirSync(outputDir, { recursive: true })) {
    if (entry.endsWith(".md")) fs.unlinkSync(path.join(outputDir, entry));
  }
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
