// Shared filesystem helpers for the docs generators.

import fs from "node:fs";
import path from "node:path";

// All files under `dir` (recursive) whose name ends with `ext`, sorted by
// their `/`-separated relative path so ordering is platform-independent.
export function listFiles(dir, ext) {
  if (!fs.existsSync(dir)) return [];
  return fs
    .readdirSync(dir, { recursive: true })
    .filter((p) => p.endsWith(ext))
    .map((p) => p.replaceAll(path.sep, "/"))
    .sort()
    .map((p) => path.join(dir, p));
}
