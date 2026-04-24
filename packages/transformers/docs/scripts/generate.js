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
  const rendered = renderModule(mod, ir, { publicNames });
  if (!hasPublicBody(rendered)) {
    console.log(`skipped ${mod.name}.md — no public content`);
    continue;
  }
  const outputPath = path.resolve(outputDir, `${mod.name}.md`);
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, rendered);
  console.log(`wrote ${mod.name}.md`);
}

// A module earns a page when its render produces either a section
// (classes/functions/constants/typedefs) or a description that links out — the
// latter covers index / entry-point modules that exist to orient the reader.
// Modules that reduce to a bare title + prose get skipped.
function hasPublicBody(markdown) {
  if (/^## /m.test(markdown)) return true;
  const body = markdown.replace(/^# [^\n]+\n/, "");
  return /\]\(/.test(body);
}

function clearExistingMarkdown() {
  if (!fs.existsSync(outputDir)) return;
  for (const entry of fs.readdirSync(outputDir, { recursive: true })) {
    if (entry.endsWith(".md")) fs.unlinkSync(path.join(outputDir, entry));
  }
}
