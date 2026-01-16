import { context } from "esbuild";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { readdirSync } from "node:fs";
import { startServer } from "../../../scripts/httpServer.mjs";
import { colors, createLogger } from "../../../scripts/logger.mjs";
import { transformAllImports } from "./transformImports.mjs";
import prepareOutDir from "../../../scripts/prepareOutDir.mjs";

const log = createLogger("react");
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.join(__dirname, "..");
const OUT_DIR = path.join(ROOT_DIR, "dist");
const SRC_DIR = path.join(ROOT_DIR, "src");
const PORT = 8081;

prepareOutDir(OUT_DIR);

const SHOULD_TRANSFORM = process.env.TRANSFORM_IMPORTS === "true";

const startTime = performance.now();

log.section("BUILD");
log.info(
  "Building @huggingface/transformers-react with esbuild in watch mode...",
);
if (SHOULD_TRANSFORM) {
  log.dim("Import transformation enabled (using http://localhost:8080)\n");
} else {
  log.dim("Import transformation disabled (using npm package)\n");
}

const transformImportsPlugin = {
  name: "transform-imports",
  setup(build) {
    build.onEnd((result) => {
      if (result.errors.length === 0 && SHOULD_TRANSFORM) {
        try {
          transformAllImports(OUT_DIR);
        } catch (error) {
          log.warning(`Import transformation failed: ${error.message}`);
        }
      }
    });
  },
};

const rebuildPlugin = {
  name: "rebuild-logger",
  setup(build) {
    let startTime = 0;
    let isFirstBuild = true;

    build.onStart(() => {
      startTime = performance.now();
      if (!isFirstBuild) {
        log.build(`${colors.gray}Rebuilding...${colors.reset}`);
      }
    });

    build.onEnd((result) => {
      const endTime = performance.now();
      const duration = (endTime - startTime).toFixed(2);

      if (result.errors.length > 0) {
        log.error(
          `Build failed with ${result.errors.length} error(s) in ${duration}ms`,
        );
      } else if (!isFirstBuild) {
        log.done(`Rebuilt in ${colors.gray}${duration}ms${colors.reset}`);
      }

      isFirstBuild = false;
    });
  },
};

// Create build context for watch mode
const buildContext = await context({
  entryPoints: [path.join(SRC_DIR, "index.ts")],
  bundle: true,
  outfile: path.join(OUT_DIR, "index.js"),
  format: "esm",
  platform: "browser",
  target: "es2020",
  sourcemap: true,
  external: ["react", "@huggingface/transformers"],
  logLevel: "silent",
  plugins: [transformImportsPlugin, rebuildPlugin],
});

log.dim("Starting initial build...\n");

// Wait for the initial build to complete
await buildContext.rebuild();

// Now start watching
await buildContext.watch();

const endTime = performance.now();
const duration = (endTime - startTime).toFixed(2);
log.success(
  `Initial build completed in ${colors.bright}${duration}ms${colors.reset}`,
);

log.section("SERVER");
const server = await startServer(OUT_DIR, PORT);

log.success(
  `Server running at ${colors.bright}http://localhost:${PORT}/${colors.reset}`,
);
log.dim(`Serving from: ${OUT_DIR}\n`);

const files = readdirSync(OUT_DIR)
  .filter((file) => !file.endsWith(".map") && !file.endsWith(".tsbuildinfo"))
  .sort();

if (files.length > 0) {
  log.info(`${colors.bright}Available files:${colors.reset}`);
  files.forEach((file) => {
    log.url(`http://localhost:${PORT}/${file}`);
  });
}

log.dim(`\nWatching for changes...\n`);

process.on("SIGINT", async () => {
  log.warning(`\nStopping watch mode and server...`);
  server.close();
  await buildContext.dispose();
  log.dim("Goodbye!");
  process.exit(0);
});
