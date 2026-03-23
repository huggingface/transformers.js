import { OUT_DIR } from "./build/constants.mjs";
import prepareOutDir from "../../../scripts/prepareOutDir.mjs";
import { colors, createLogger } from "../../../scripts/logger.mjs";
import { buildAll } from "./build/buildAll.mjs";

const log = createLogger("transformers");

log.section("BUILD");
log.info("Building transformers.js with esbuild...\n");

const startTime = performance.now();

try {
  prepareOutDir(OUT_DIR);

  await buildAll(log);

  const endTime = performance.now();
  const duration = (endTime - startTime).toFixed(2);
  log.success(`All builds completed in ${colors.bright}${duration}ms${colors.reset}\n`);
} catch (error) {
  log.error(`Build failed: ${error.message}`);
  console.error(error);
  process.exit(1);
}
