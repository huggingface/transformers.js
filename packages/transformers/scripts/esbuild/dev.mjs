import { context } from "esbuild";
import path from "node:path";
import { readdirSync } from "node:fs";
import { postBuildPlugin } from "./build/plugins/postBuildPlugin.mjs";
import { stripNodePrefixPlugin } from "./build/plugins/stripNodePrefixPlugin.mjs";
import { ignoreModulesPlugin } from "./build/plugins/ignoreModulesPlugin.mjs";
import { rebuildPlugin } from "./build/plugins/rebuildPlugin.mjs";
import { externalNodeBuiltinsPlugin } from "./build/plugins/externalNodeBuiltinsPlugin.mjs";
import {
  getEsbuildDevConfig,
  OUT_DIR,
  ROOT_DIR,
  WEB_IGNORE_MODULES,
} from "./build/constants.mjs";
import { startServer } from "../../../../scripts/httpServer.mjs";
import prepareOutDir from "../../../../scripts/prepareOutDir.mjs";
import { colors, createLogger } from "../../../../scripts/logger.mjs";

const log = createLogger("transformers");
const startTime = performance.now();

prepareOutDir(OUT_DIR);

log.section("BUILD");
log.info("Building transformers.js with esbuild in watch mode...");

// Create build contexts for watch mode
const bundleContext = await context({
  ...getEsbuildDevConfig(ROOT_DIR),
  outfile: path.join(OUT_DIR, "transformers.js"),
  plugins: [
    ignoreModulesPlugin(WEB_IGNORE_MODULES),
    stripNodePrefixPlugin(),
    externalNodeBuiltinsPlugin(),
    postBuildPlugin(OUT_DIR, ROOT_DIR),
    rebuildPlugin("Bundle"),
  ],
});

const webContext = await context({
  ...getEsbuildDevConfig(ROOT_DIR),
  outfile: path.join(OUT_DIR, "transformers.web.js"),
  external: ["onnxruntime-common", "onnxruntime-web"],
  plugins: [
    ignoreModulesPlugin(WEB_IGNORE_MODULES),
    stripNodePrefixPlugin(),
    externalNodeBuiltinsPlugin(),
    rebuildPlugin("Web"),
  ],
});

log.dim("Starting initial build...\n");

// Wait for the initial builds to complete before starting the server
await Promise.all([bundleContext.rebuild(), webContext.rebuild()]);

// Now start watching
await Promise.all([bundleContext.watch(), webContext.watch()]);

const endTime = performance.now();
const duration = (endTime - startTime).toFixed(2);
log.success(
  `All builds completed in ${colors.bright}${duration}ms${colors.reset}`,
);

const PORT = 8080;

log.section("SERVER");
const server = await startServer(OUT_DIR, PORT);

log.success(
  `Server running at ${colors.bright}http://localhost:${PORT}/${colors.reset}`,
);
log.dim(`Serving from: ${OUT_DIR}\n`);

// List all files in OUT_DIR (excluding .map files)
const files = readdirSync(OUT_DIR)
  .filter((file) => !file.endsWith(".map"))
  .sort();

if (files.length > 0) {
  log.info(`${colors.bright}Available files:${colors.reset}`);
  files.forEach((file) => {
    log.url(`http://localhost:${PORT}/${file}`);
  });
}

log.dim(`\nWatching for changes...\n`);

// Keep process alive and cleanup
process.on("SIGINT", async () => {
  log.warning(`\nStopping watch mode and server...`);
  server.close();
  await bundleContext.dispose();
  await webContext.dispose();
  log.dim("Goodbye!");
  process.exit(0);
});
