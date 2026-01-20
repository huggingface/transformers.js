import { context } from "esbuild";
import path from "node:path";
import { OUT_DIR, getEsbuildDevConfig } from "./constants.mjs";
import { rebuildPlugin } from "../../../../../scripts/rebuildPlugin.mjs";
import { BUILD_TARGETS } from "./targets.mjs";

/**
 * Create an esbuild context for a single build target
 */
async function createBuildContext(targetName, targetConfig, log) {
  const { outputFile, platform } = targetConfig;
  const config = getEsbuildDevConfig();

  return context({
    ...config,
    outfile: path.join(OUT_DIR, outputFile),
    platform,
    plugins: [rebuildPlugin(targetName, log, { logFirstBuild: false })],
  });
}

/**
 * Build all targets in watch mode for development
 * Only builds targets that have includeInDevMode set to true
 * @returns {Promise<Array>} Array of esbuild contexts that can be disposed later
 */
export async function buildAllWithWatch(log) {
  log.dim("Creating build contexts...\n");

  // Filter targets for dev mode
  const devTargets = BUILD_TARGETS.filter((t) => t.config.includeInDevMode);

  // Create contexts for dev targets
  const contexts = await Promise.all(
    devTargets.map((target) =>
      createBuildContext(target.name, target.config, log),
    ),
  );

  log.dim("Starting initial build...\n");

  // Wait for the initial builds to complete
  await Promise.all(contexts.map((ctx) => ctx.rebuild()));

  // Start watching all targets
  await Promise.all(contexts.map((ctx) => ctx.watch()));

  return contexts;
}
