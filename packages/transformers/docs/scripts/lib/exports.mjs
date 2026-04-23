// Resolve the symbol names the library exposes via `src/transformers.js`.
// Everything outside this set is treated as internal and excluded from docs.

import fs from "node:fs";
import path from "node:path";

const STAR_REEXPORT = /export\s+\*\s+from\s+['"]([^'"]+)['"]/g;
const NAMED_EXPORT = /export\s*\{\s*([^}]+)\s*\}\s*(?:from\s*['"][^'"]+['"])?/g;
const DECL_EXPORT = /export\s+(?:default\s+)?(?:async\s+)?(?:class|function\s*\*?|const|let|var)\s+([A-Za-z_$][\w$]*)/g;

export function collectPublicExports(entryFile) {
  const names = new Set();
  walk(path.resolve(entryFile), new Set(), names);
  return names;
}

function walk(file, visited, names) {
  if (visited.has(file) || !fs.existsSync(file)) return;
  visited.add(file);

  const source = stripComments(fs.readFileSync(file, "utf8"));
  const dir = path.dirname(file);

  for (const [, specifier] of source.matchAll(STAR_REEXPORT)) {
    if (!specifier.startsWith(".")) continue;
    walk(path.resolve(dir, specifier), visited, names);
  }

  for (const [, list] of source.matchAll(NAMED_EXPORT)) {
    for (const spec of list.split(",")) {
      const parts = spec.trim().split(/\s+as\s+/);
      const exported = (parts[1] ?? parts[0]).trim();
      if (exported) names.add(exported);
    }
  }

  for (const [, name] of source.matchAll(DECL_EXPORT)) names.add(name);
}

function stripComments(source) {
  return source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/(^|\s)\/\/[^\n]*/g, "$1");
}
