import { context } from "esbuild";
import path from "node:path";
import { postBuildPlugin } from "./build/plugins/postBuildPlugin.mjs";
import { stripNodePrefixPlugin } from "./build/plugins/stripNodePrefixPlugin.mjs";
import { ignoreModulesPlugin } from "./build/plugins/ignoreModulesPlugin.mjs";
import { rebuildPlugin } from "./build/plugins/rebuildPlugin.mjs";
import { externalNodeBuiltinsPlugin } from "./build/plugins/externalNodeBuiltinsPlugin.mjs";
import { getEsbuildDevConfig, OUT_DIR, ROOT_DIR, WEB_IGNORE_MODULES } from "./build/constants.mjs";
import { startServer } from "./build/httpServer.mjs";
import prepareOutDir from "./build/prepareOutDir.mjs";

const startTime = performance.now();

prepareOutDir(OUT_DIR);

console.log("\n=== BUILD ===");
console.log("Building transformers.js with esbuild in watch mode...");

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

console.log("\nInitial build starting...");

await Promise.all([bundleContext.watch(), webContext.watch()]);

const endTime = performance.now();
const duration = (endTime - startTime).toFixed(2);
console.log(`\nAll builds completed successfully in ${duration}ms!`);

const PORT = 8080;

console.log("\n=== SERVE ===");
const server = await startServer(OUT_DIR, PORT);

console.log(`\nServer running at http://localhost:${PORT}/`);
console.log(`Serving files from: ${OUT_DIR}`);

console.log(`\nWatching for changes...\n`);

// Keep process alive and cleanup
process.on("SIGINT", async () => {
  console.log("\n\nStopping watch mode and server...");
  server.close();
  await bundleContext.dispose();
  await webContext.dispose();
  process.exit(0);
});
