import { OUT_DIR } from "./build/constants.mjs";
import prepareOutDir from "../../../../scripts/prepareOutDir.mjs";
import { colors, createLogger } from "../../../../scripts/logger.mjs";
import { buildAllWithWatch } from "./build/buildAllWithWatch.mjs";

const log = createLogger("transformers");
const startTime = performance.now();

prepareOutDir(OUT_DIR);

log.section("BUILD");
log.info("Building transformers.js with esbuild in watch mode...");

// Build all targets with watch mode
const contexts = await buildAllWithWatch(log);

const endTime = performance.now();
const duration = (endTime - startTime).toFixed(2);
log.success(
  `All builds completed in ${colors.bright}${duration}ms${colors.reset}`,
);

log.dim(`\nWatching for changes...\n`);

// Keep process alive and cleanup
process.on("SIGINT", async () => {
  log.warning(`\nStopping watch mode...`);
  await Promise.all(contexts.map((ctx) => ctx.dispose()));
  log.dim("Goodbye!");
  process.exit(0);
});
