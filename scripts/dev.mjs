import { context } from "esbuild";
import { existsSync, mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { postBuildPlugin } from "./build/esbuild/postBuildPlugin.mjs";
import { stripNodePrefixPlugin } from "./build/esbuild/stripNodePrefixPlugin.mjs";
import { ignoreModulesPlugin } from "./build/esbuild/ignoreModulesPlugin.mjs";
import { rebuildPlugin } from "./build/esbuild/rebuildPlugin.mjs";
import { DIST_FOLDER, WEB_IGNORE_MODULES } from "./build/constants.mjs";
import { startServer } from "./build/httpServer.mjs";
import { execSync } from "node:child_process";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.join(__dirname, "..");

const outdir = path.join(rootDir, DIST_FOLDER);
const startTime = performance.now();

// Ensure output directory exists
if (!existsSync(outdir)) {
  mkdirSync(outdir, { recursive: true });
}

console.log("=== CLEAN ===");
execSync(`rimraf ${outdir}`, { stdio: "inherit" });
console.log(`cleaned directory: ${outdir}`)

console.log("\n=== BUILD ===");
console.log("Building transformers.js with esbuild in watch mode...");

// Create build contexts for watch mode
const bundleContext = await context({
  bundle: true,
  treeShaking: true,
  logLevel: "info",
  entryPoints: [path.join(rootDir, "src/transformers.js")],
  platform: "neutral",
  format: "esm",
  outfile: path.join(outdir, "transformers.js"),
  sourcemap: true,
  plugins: [
    ignoreModulesPlugin(WEB_IGNORE_MODULES),
    stripNodePrefixPlugin(),
    postBuildPlugin(outdir, rootDir),
    rebuildPlugin("Bundle"),
  ],
  logOverride: {
    "empty-import-meta": "silent",
  },
});

const webContext = await context({
  bundle: true,
  treeShaking: true,
  logLevel: "info",
  entryPoints: [path.join(rootDir, "src/transformers.js")],
  platform: "neutral",
  format: "esm",
  outfile: path.join(outdir, "transformers.web.js"),
  sourcemap: true,
  external: ["onnxruntime-common", "onnxruntime-web"],
  plugins: [
    ignoreModulesPlugin(WEB_IGNORE_MODULES),
    stripNodePrefixPlugin(),
    rebuildPlugin("Web"),
  ],
  logOverride: {
    "empty-import-meta": "silent",
  },
});

console.log("\nInitial build starting...");

await Promise.all([bundleContext.watch(), webContext.watch()]);

const endTime = performance.now();
const duration = (endTime - startTime).toFixed(2);
console.log(`\nAll builds completed successfully in ${duration}ms!`);

const PORT = 8080;

console.log("\n=== SERVE ===");
const server = await startServer(outdir, PORT);

console.log(`\nServer running at http://localhost:${PORT}/`);
console.log(`Serving files from: ${outdir}`);

console.log(`\nWatching for changes...\n`);

// Keep process alive and cleanup
process.on("SIGINT", async () => {
  console.log("\n\nStopping watch mode and server...");
  server.close();
  await bundleContext.dispose();
  await webContext.dispose();
  process.exit(0);
});
