import fs from "node:fs";
import path from "node:path";

import { apiOutputDir, toctreePath } from "./paths.mjs";

export function validateGeneratedDocs({ outputDir = apiOutputDir, tocPath = toctreePath } = {}) {
  const generatedApiPages = listApiPages(outputDir);
  const linkedApiPages = readToctreeApiPages(tocPath);

  const unlisted = difference(generatedApiPages, linkedApiPages);
  const stale = difference(linkedApiPages, generatedApiPages);

  return {
    ok: unlisted.length === 0 && stale.length === 0,
    unlisted,
    stale,
  };
}

export function formatValidationResult(result) {
  if (result.ok) return "validated docs toctree";

  const lines = ["docs validation failed"];
  if (result.unlisted.length) {
    lines.push("", "Generated API pages missing from _toctree.yml:");
    for (const page of result.unlisted) lines.push(`- ${page}`);
  }
  if (result.stale.length) {
    lines.push("", "API pages listed in _toctree.yml but not generated:");
    for (const page of result.stale) lines.push(`- ${page}`);
  }
  return lines.join("\n");
}

function listApiPages(outputDir) {
  return listMarkdown(outputDir)
    .map((file) => toApiPage(outputDir, file))
    .sort();
}

function listMarkdown(dir) {
  if (!fs.existsSync(dir)) return [];

  const out = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out.push(...listMarkdown(full));
    else if (entry.name.endsWith(".md")) out.push(full);
  }
  return out;
}

function toApiPage(outputDir, file) {
  const relative = path.relative(outputDir, file).replaceAll(path.sep, "/").replace(/\.md$/, "");
  return `api/${relative}`;
}

function readToctreeApiPages(tocPath) {
  if (!fs.existsSync(tocPath)) return [];

  const pages = [];
  const re = /^\s*-\s+local:\s+(api\/\S+)\s*$/gm;
  const text = fs.readFileSync(tocPath, "utf8");
  for (const match of text.matchAll(re)) pages.push(match[1]);
  return pages.sort();
}

function difference(a, b) {
  const right = new Set(b);
  return a.filter((item) => !right.has(item));
}
