import { colors, createLogger } from "../../../scripts/logger.mjs";
import prepareOutDir from "../../../scripts/prepareOutDir.mjs";
import { OUT_DIR } from "./esbuild/build/constants.mjs";
import { buildAllWithWatch } from "./esbuild/build/buildAllWithWatch.mjs";

const log = createLogger("react");

prepareOutDir(OUT_DIR);

const shouldTransform = process.env.TRANSFORM_IMPORTS === "true";

const startTime = performance.now();

log.section("BUILD");
log.info(
  "Building @huggingface/transformers-react with esbuild in watch mode...",
);
if (shouldTransform) {
  log.dim("Import transformation enabled (using http://localhost:8080)\n");
} else {
  log.dim("Import transformation disabled (using npm package)\n");
}

// Build all targets with watch mode
const contexts = await buildAllWithWatch(log);

const endTime = performance.now();
const duration = (endTime - startTime).toFixed(2);
log.success(
  `Initial build completed in ${colors.bright}${duration}ms${colors.reset}`,
);

log.dim(`\nWatching for changes...\n`);

process.on("SIGINT", async () => {
  log.warning(`\nStopping watch mode...`);
  await Promise.all(contexts.map((ctx) => ctx.dispose()));
  log.dim("Goodbye!");
  process.exit(0);
});
