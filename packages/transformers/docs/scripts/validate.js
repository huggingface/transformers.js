#!/usr/bin/env node

import { formatValidationResult, validateGeneratedDocs } from "./lib/validate.mjs";

const validation = validateGeneratedDocs();
console.log(formatValidationResult(validation));
if (!validation.ok) process.exitCode = 1;
