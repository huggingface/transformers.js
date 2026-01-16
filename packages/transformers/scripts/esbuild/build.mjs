import { build as esbuild } from "esbuild";
import path from "node:path";
import { stripNodePrefixPlugin } from "./build/plugins/stripNodePrefixPlugin.mjs";
import { ignoreModulesPlugin } from "./build/plugins/ignoreModulesPlugin.mjs";
import { postBuildPlugin } from "./build/plugins/postBuildPlugin.mjs";
import { externalNodeBuiltinsPlugin } from "./build/plugins/externalNodeBuiltinsPlugin.mjs";
import {
  NODE_IGNORE_MODULES,
  NODE_EXTERNAL_MODULES,
  WEB_IGNORE_MODULES,
  WEB_EXTERNAL_MODULES,
  OUT_DIR,
  ROOT_DIR,
  getEsbuildProdConfig,
} from "./build/constants.mjs";
import { reportSize } from "./build/reportSize.mjs";
import prepareOutDir from "./build/prepareOutDir.mjs";
import { colors, log } from "../../../../scripts/logger.mjs";

/**
 *
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

  const regularFile = `transformers${name}${suffix}`;
  const minFile = `transformers${name}.min${suffix}`;

  const plugins = [];
  // Add ignoreModulesPlugin FIRST so it can catch modules before stripNodePrefixPlugin marks them as external
  if (ignoreModules.length > 0) {
    plugins.push(ignoreModulesPlugin(ignoreModules));
  }
  plugins.push(stripNodePrefixPlugin());
  plugins.push(externalNodeBuiltinsPlugin());
  if (usePostBuild) {
    plugins.push(postBuildPlugin(OUT_DIR, ROOT_DIR));
  }

  log.build(`Building ${colors.bright}${regularFile}${colors.reset}...`);
  await esbuild({
    ...getEsbuildProdConfig(ROOT_DIR),
    platform,
    format,
    outfile: path.join(OUT_DIR, regularFile),
    external: externalModules,
    plugins,
  });
  reportSize(path.join(OUT_DIR, regularFile));

  log.build(`Building ${colors.bright}${minFile}${colors.reset}...`);
  await esbuild({
    ...getEsbuildProdConfig(ROOT_DIR),
    platform,
    format,
    outfile: path.join(OUT_DIR, minFile),
    minify: true,
    external: externalModules,
    plugins,
    legalComments: "none",
  });
  reportSize(path.join(OUT_DIR, minFile));
}

log.section("BUILD");
log.info("Building transformers.js with esbuild...\n");

const startTime = performance.now();

try {
  prepareOutDir(OUT_DIR);

  // Bundle build - bundles everything except ignored modules
  log.section("Bundle Build (ESM)");
  await buildTarget({
    name: "",
    suffix: ".js",
    format: "esm",
    ignoreModules: WEB_IGNORE_MODULES,
    externalModules: [],
    usePostBuild: true,
  });

  // Web build - external onnxruntime libs
  log.section("Web Build (ESM)");
  await buildTarget({
    name: ".web",
    suffix: ".js",
    format: "esm",
    ignoreModules: WEB_IGNORE_MODULES,
    externalModules: WEB_EXTERNAL_MODULES,
    usePostBuild: false,
  });

  // Node ESM build
  log.section("Node Build (ESM)");
  await buildTarget({
    name: ".node",
    suffix: ".mjs",
    format: "esm",
    ignoreModules: NODE_IGNORE_MODULES,
    externalModules: NODE_EXTERNAL_MODULES,
    usePostBuild: false,
  });

  // Node CJS build
  log.section("Node Build (CJS)");
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
  log.success(
    `All builds completed in ${colors.bright}${duration}ms${colors.reset}\n`,
  );
} catch (error) {
  log.error(`Build failed: ${error.message}`);
  console.error(error);
  process.exit(1);
}
