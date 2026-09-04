import path from "node:path";
import url from "node:url";

const libDir = path.dirname(url.fileURLToPath(import.meta.url));

export const docsDir = path.resolve(libDir, "..", "..");
export const packageRoot = path.dirname(docsDir);
export const repoRoot = path.resolve(packageRoot, "..", "..");

export const apiOutputDir = path.join(docsDir, "source", "api");
export const toctreePath = path.join(docsDir, "source", "_toctree.yml");
export const skillDir = path.join(repoRoot, ".ai", "skills", "transformers-js");
