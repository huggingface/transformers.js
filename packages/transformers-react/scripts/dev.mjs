import { spawn } from "node:child_process";
import { unlinkSync } from "node:fs";
import { colors, createLogger } from "../../../scripts/logger.mjs";
import prepareOutDir from "../../../scripts/prepareOutDir.mjs";
import { OUT_DIR, ROOT_DIR } from "./esbuild/build/constants.mjs";
import { buildAllWithWatch } from "./esbuild/build/buildAllWithWatch.mjs";

const log = createLogger("react");

prepareOutDir(OUT_DIR);

// Remove tsbuildinfo to force TypeScript to rebuild type declarations
try {
  unlinkSync(`${ROOT_DIR}/tsconfig.tsbuildinfo`);
} catch (err) {
  // File doesn't exist, that's fine
}

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
        line.includes("Found") && line.includes("error")
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

process.on("SIGINT", async () => {
  log.warning(`\nStopping watch mode...`);
  tscWatch.kill();
  await Promise.all(contexts.map((ctx) => ctx.dispose()));
  log.dim("Goodbye!");
  process.exit(0);
});
