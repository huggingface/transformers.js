#!/usr/bin/env node

// Validate the generated docs without regenerating them: a CI check that the gitignored
// `docs/source/api/` markdown matches `docs/source/_toctree.yml` and that internal links resolve.

import { listFiles } from "./lib/fs.mjs";
import { apiOutputDir } from "./lib/paths.mjs";
import { formatValidationResult, validateGeneratedDocs } from "./lib/validate.mjs";

if (!listFiles(apiOutputDir, ".md").length) {
  console.log(`no generated api markdown in ${apiOutputDir} — run \`pnpm docs-generate\` first`);
}

const validation = validateGeneratedDocs();
console.log(formatValidationResult(validation));
if (!validation.ok) process.exitCode = 1;
