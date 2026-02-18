import { context } from "esbuild";
import path from "node:path";
import { stripNodePrefixPlugin } from "./plugins/stripNodePrefixPlugin.mjs";
import { ignoreModulesPlugin } from "./plugins/ignoreModulesPlugin.mjs";
import { postBuildPlugin } from "./plugins/postBuildPlugin.mjs";
import { externalNodeBuiltinsPlugin } from "./plugins/externalNodeBuiltinsPlugin.mjs";
import { rebuildPlugin } from "../../../../scripts/rebuildPlugin.mjs";
import { OUT_DIR, ROOT_DIR, getEsbuildDevConfig } from "./constants.mjs";
import { BUILD_TARGETS } from "./targets.mjs";

/**
 * Create an esbuild context for a single build target
 */
async function createBuildContext(targetName, targetConfig, log) {
  const { name, suffix, format, ignoreModules, externalModules, usePostBuild } = targetConfig;
  const platform = format === "cjs" ? "node" : "neutral";
  const outputFile = `transformers${name}${suffix}`;

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
  plugins.push(rebuildPlugin(targetName, log));

  return context({
    ...getEsbuildDevConfig(ROOT_DIR),
    platform,
    format,
    outfile: path.join(OUT_DIR, outputFile),
    external: externalModules,
    plugins,
  });
}

/**
 * Build all targets in watch mode for development
 * @returns {Promise<Array>} Array of esbuild contexts that can be disposed later
 */
export async function buildAllWithWatch(log) {
  log.dim("Creating build contexts...\n");

  // Create contexts for all targets
  const contexts = await Promise.all(BUILD_TARGETS.map((target) => createBuildContext(target.name, target.config, log)));

  log.dim("Starting initial build...\n");

  // Wait for the initial builds to complete
  await Promise.all(contexts.map((ctx) => ctx.rebuild()));

  // Start watching all targets
  await Promise.all(contexts.map((ctx) => ctx.watch()));

  return contexts;
}
