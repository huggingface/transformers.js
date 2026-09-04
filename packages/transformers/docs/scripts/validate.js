#!/usr/bin/env node

// Validate the generated docs without regenerating them. Useful in CI as a
// fast check that the committed `docs/source/api/` markdown is consistent
// with `docs/source/_toctree.yml` and that internal links resolve.

import { formatValidationResult, validateGeneratedDocs } from "./lib/validate.mjs";

const validation = validateGeneratedDocs();
console.log(formatValidationResult(validation));
if (!validation.ok) process.exitCode = 1;
