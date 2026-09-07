#!/usr/bin/env node

import path from "node:path";

import { buildReadme } from "./build_readme.js";
import { generateApiDocs } from "./lib/generate-api.mjs";
import { loadProject } from "./lib/load.mjs";
import { packageRoot, repoRoot, skillDir } from "./lib/paths.mjs";
import { renderSkill } from "./lib/render-skill.mjs";
import { formatValidationResult, validateGeneratedDocs } from "./lib/validate.mjs";

const project = loadProject(packageRoot);

// Run every phase even if an earlier one fails, so a single run surfaces all
// problems; collect errors per phase and fail at the end.
const errors = [];
const runPhase = (name, fn) => {
  try {
    return fn();
  } catch (err) {
    errors.push(`${name}: ${err.message}`);
    return null;
  }
};

const apiResult = runPhase("api docs", () => generateApiDocs({ project }));
for (const err of apiResult?.errors ?? []) errors.push(`api docs: ${err}`);

const skillResult = runPhase("skill", () => {
  const result = renderSkill({ ir: project.ir, tasks: project.tasks, publicNames: project.publicNames, skillDir });
  console.log(`wrote skill to ${path.relative(repoRoot, skillDir) || skillDir}`);
  return result;
});
for (const err of skillResult?.errors ?? []) errors.push(`skill: ${err}`);

runPhase("readme", () => {
  const readmePath = buildReadme({ project });
  console.log(`wrote ${path.relative(process.cwd(), readmePath)}`);
});

const validation = runPhase("validation", () => validateGeneratedDocs({ project }));
if (validation) console.log(formatValidationResult(validation));

if (errors.length) {
  console.log("");
  console.log(`docs generation failed with ${errors.length} error${errors.length === 1 ? "" : "s"}:`);
  for (const err of errors) console.log(`- ${err}`);
}
if (errors.length || !validation?.ok) process.exitCode = 1;
