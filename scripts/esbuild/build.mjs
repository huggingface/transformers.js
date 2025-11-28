import { build as esbuild } from "esbuild";
import { execSync } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { stripNodePrefixPlugin } from "./build/plugins/stripNodePrefixPlugin.mjs";
import { ignoreModulesPlugin } from "./build/plugins/ignoreModulesPlugin.mjs";
import { postBuildPlugin } from "./build/plugins/postBuildPlugin.mjs";
import { DIST_FOLDER, NODE_IGNORE_MODULES, NODE_EXTERNAL_MODULES, WEB_IGNORE_MODULES, WEB_EXTERNAL_MODULES } from "./build/constants.mjs";
import { reportSize } from "./build/reportSize.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.join(__dirname, "../..");

/**
 * Helper function to create build configurations.
 * Equivalent to webpack's buildConfig function.
 */
async function buildTarget({
  name = "",
  suffix = ".js",
  format = "esm", // 'esm' | 'cjs'
  ignoreModules = [],
  externalModules = [],
  usePostBuild = false,
}) {
  const platform = format === "cjs" ? "node" : "neutral";
  const outdir = path.join(rootDir, DIST_FOLDER);

  if (!existsSync(outdir)) {
    mkdirSync(outdir, { recursive: true });
  }

  const regularFile = `transformers${name}${suffix}`;
  const minFile = `transformers${name}.min${suffix}`;

  const plugins = [];
  // Add ignoreModulesPlugin FIRST so it can catch modules before stripNodePrefixPlugin marks them as external
  if (ignoreModules.length > 0) {
    plugins.push(ignoreModulesPlugin(ignoreModules));
  }
  plugins.push(stripNodePrefixPlugin());
  if (usePostBuild) {
    plugins.push(postBuildPlugin(outdir, rootDir));
  }

  console.log(`\nBuilding ${regularFile}...`);
  await esbuild({
    bundle: true,
    treeShaking: true,
    logLevel: "warning",
    entryPoints: [path.join(rootDir, "src/transformers.js")],
    platform,
    format,
    outfile: path.join(outdir, regularFile),
    sourcemap: true,
    external: externalModules,
    plugins,
    logOverride: {
      // Suppress import.meta warning for CJS builds - it's handled gracefully in the code
      "empty-import-meta": "silent",
    },
  });
  reportSize(path.join(outdir, regularFile));

  console.log(`\nBuilding ${minFile}...`);
  await esbuild({
    bundle: true,
    treeShaking: true,
    logLevel: "warning",
    entryPoints: [path.join(rootDir, "src/transformers.js")],
    platform,
    format,
    outfile: path.join(outdir, minFile),
    sourcemap: true,
    minify: true,
    external: externalModules,
    plugins,
    legalComments: "none",
    logOverride: {
      // Suppress import.meta warning for CJS builds - it's handled gracefully in the code
      "empty-import-meta": "silent",
    },
  });
  reportSize(path.join(outdir, minFile));
}

console.log("\nBuilding transformers.js with esbuild...\n");

const startTime = performance.now();

try {
  console.log("=== CLEAN ===");
  execSync(`rimraf ${DIST_FOLDER}`, { stdio: "inherit" });

  // Bundle build - bundles everything except ignored modules
  console.log("\n=== Bundle Build (ESM) ===");
  await buildTarget({
    name: "",
    suffix: ".js",
    format: "esm",
    ignoreModules: WEB_IGNORE_MODULES,
    externalModules: [],
    usePostBuild: true,
  });

  // Web build - external onnxruntime libs
  console.log("\n=== Web Build (ESM) ===");
  await buildTarget({
    name: ".web",
    suffix: ".js",
    format: "esm",
    ignoreModules: WEB_IGNORE_MODULES,
    externalModules: WEB_EXTERNAL_MODULES,
    usePostBuild: false,
  });

  // Node ESM build
  console.log("\n=== Node Build (ESM) ===");
  await buildTarget({
    name: ".node",
    suffix: ".mjs",
    format: "esm",
    ignoreModules: NODE_IGNORE_MODULES,
    externalModules: NODE_EXTERNAL_MODULES,
    usePostBuild: false,
  });

  // Node CJS build
  console.log("\n=== Node Build (CJS) ===");
  await buildTarget({
    name: ".node",
    suffix: ".cjs",
    format: "cjs",
    ignoreModules: NODE_IGNORE_MODULES,
    externalModules: NODE_EXTERNAL_MODULES,
    usePostBuild: false,
  });

  const endTime = performance.now();
  const duration = (endTime - startTime).toFixed(2);
  console.log(`\nAll builds completed successfully in ${duration}ms!\n`);
} catch (error) {
  console.error("\nBuild failed:", error);
  process.exit(1);
}
