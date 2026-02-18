import { spawn } from "node:child_process";
import { unlinkSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { OUT_DIR } from "./build/constants.mjs";
import prepareOutDir from "../../../scripts/prepareOutDir.mjs";
import { colors, createLogger } from "../../../scripts/logger.mjs";
import { buildAllWithWatch } from "./build/buildAllWithWatch.mjs";

const log = createLogger("transformers");
const startTime = performance.now();

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.join(__dirname, "..");

prepareOutDir(OUT_DIR);

// Remove tsbuildinfo to force TypeScript to rebuild type declarations
try {
  unlinkSync(`${ROOT_DIR}/types/tsconfig.tsbuildinfo`);
} catch (err) {
  // File doesn't exist, that's fine
}

log.section("BUILD");
log.info("Building transformers.js with esbuild in watch mode...");

// Build all targets with watch mode
const contexts = await buildAllWithWatch(log);

const endTime = performance.now();
const duration = (endTime - startTime).toFixed(2);
log.success(`All builds completed in ${colors.bright}${duration}ms${colors.reset}`);

// Generate initial TypeScript declarations, then start watch mode
log.section("TYPES");
log.info("Generating initial type declarations...");

await new Promise((resolve, reject) => {
  const tscBuild = spawn("tsc", ["--build"], {
    cwd: ROOT_DIR,
    stdio: "pipe",
    shell: true,
  });

  tscBuild.stdout.on("data", (data) => {
    const output = data.toString().trim();
    if (output && output.includes("error")) {
      log.dim(output);
    }
  });

  tscBuild.stderr.on("data", (data) => {
    const output = data.toString().trim();
    if (output) {
      log.error(output);
    }
  });

  tscBuild.on("exit", (code) => {
    if (code === 0) {
      log.done("Type declarations generated");
      resolve();
    } else {
      reject(new Error(`TypeScript build failed with code ${code}`));
    }
  });
});

log.info("Starting TypeScript watch mode...\n");

const tscWatch = spawn("tsc", ["--build", "--watch", "--preserveWatchOutput"], {
  cwd: ROOT_DIR,
  stdio: "pipe",
  shell: true,
});

tscWatch.stdout.on("data", (data) => {
  const output = data.toString().trim();
  if (output) {
    output.split("\n").forEach((line) => {
      // Filter out verbose output, only show important messages
      if (
        line.includes("error") ||
        line.includes("File change detected") ||
        line.includes("Found 0 errors") ||
        (line.includes("Found") && line.includes("error"))
      ) {
        log.dim(line);
      }
    });
  }
});

tscWatch.stderr.on("data", (data) => {
  const output = data.toString().trim();
  if (output) {
    output.split("\n").forEach((line) => {
      log.error(line);
    });
  }
});

log.dim(`Watching for changes...\n`);

// Keep process alive and cleanup
process.on("SIGINT", async () => {
  log.warning(`\nStopping watch mode...`);
  tscWatch.kill();
  await Promise.all(contexts.map((ctx) => ctx.dispose()));
  log.dim("Goodbye!");
  process.exit(0);
});
