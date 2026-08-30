import { context } from "esbuild";
import path from "node:path";
import { rebuildPlugin } from "../../../../scripts/rebuildPlugin.mjs";
import { OUT_DIR, ROOT_DIR, getEsbuildDevConfig } from "./constants.mjs";
import { BUILD_TARGETS } from "./targets.mjs";

/**
 * Create an esbuild context for a single build target
 */
async function createBuildContext(targetName, targetConfig, log) {
  const { name, suffix, format, externalModules } = targetConfig;
  const outputFile = `index${name}${suffix}`;

  const plugins = [rebuildPlugin(targetName, log)];

  return context({
    ...getEsbuildDevConfig(ROOT_DIR),
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
