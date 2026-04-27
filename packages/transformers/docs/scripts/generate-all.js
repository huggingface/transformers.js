#!/usr/bin/env node

import { generateApiDocs } from "./lib/generate-api.mjs";
import { generateSkillDocs } from "./lib/generate-skill.mjs";
import { loadProject } from "./lib/load.mjs";
import { packageRoot } from "./lib/paths.mjs";
import { formatValidationResult, validateGeneratedDocs } from "./lib/validate.mjs";

const project = loadProject(packageRoot);

generateApiDocs({ project });
generateSkillDocs({ project });

const validation = validateGeneratedDocs();
console.log(formatValidationResult(validation));
if (!validation.ok) process.exitCode = 1;
