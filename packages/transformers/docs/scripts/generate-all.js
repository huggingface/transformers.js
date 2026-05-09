#!/usr/bin/env node

import path from "node:path";

import { buildReadme } from "./build_readme.js";
import { generateApiDocs } from "./lib/generate-api.mjs";
import { generateSkillDocs } from "./lib/generate-skill.mjs";
import { loadProject } from "./lib/load.mjs";
import { packageRoot } from "./lib/paths.mjs";
import { formatValidationResult, validateGeneratedDocs } from "./lib/validate.mjs";

const project = loadProject(packageRoot);

generateApiDocs({ project });
generateSkillDocs({ project });
const readmePath = buildReadme({ project });
console.log(`wrote ${path.relative(process.cwd(), readmePath)}`);

const validation = validateGeneratedDocs({ project });
console.log(formatValidationResult(validation));
if (!validation.ok) process.exitCode = 1;
