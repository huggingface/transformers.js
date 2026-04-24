// Generate the `.ai/skills/transformers-js/` tree from the library's JSDoc.
// Injects generated task recipes into the hand-written SKILL.md (preserving
// editorial prose) and rewrites the fully-generated reference files.

import path from "node:path";
import url from "node:url";

import { loadProject } from "./lib/load.mjs";
import { renderSkill } from "./lib/render-skill.mjs";

const docs = path.dirname(path.dirname(url.fileURLToPath(import.meta.url)));
const packageRoot = path.dirname(docs);
const repoRoot = path.resolve(packageRoot, "..", "..");
const skillDir = path.join(repoRoot, ".ai", "skills", "transformers-js");

const { ir, tasks, publicNames } = loadProject(packageRoot);

renderSkill({ ir, tasks, publicNames, skillDir });

console.log(`wrote skill to ${path.relative(repoRoot, skillDir) || skillDir}`);
