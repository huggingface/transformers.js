import { spawnSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const corpusRoot = process.env.LLGUIDANCE_JS_PATH ?? resolve(here, "../../../../../llguidance-js");
const files = ["test/upstream/json-primitives.test.mjs", "test/upstream/json-structures.test.mjs", "test/upstream/json-combinations.test.mjs", "test/upstream/json-string-format.test.mjs", "test/upstream/json-x-guidance.test.mjs", "test/json-schema-2020-profile.test.mjs"].map((file) => resolve(corpusRoot, file));
const original = process.argv.includes("--original");
const unsupported = ["resolves general local JSON Pointers, escaped segments, and nested defs", "rejects malformed shapes for the production profile at compile time", "flexible separators with spaces: pretty_print$", "flexible separators with spaces: pretty_print_2$"].join("|");
const args = [...(original ? [] : ["--experimental-loader", resolve(here, "loader.mjs"), "--test-skip-pattern", unsupported]), "--test", ...files];
const result = spawnSync(process.execPath, args, { stdio: "inherit" });
process.exit(result.status ?? 1);
