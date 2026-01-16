import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { readdirSync } from "node:fs";
import { startServer } from "../../../scripts/httpServer.mjs";
import { colors, log } from "../../../scripts/logger.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.join(__dirname, "..");
const OUT_DIR = path.join(ROOT_DIR, "dist");
const PORT = 8081; // Use different port from transformers

const startTime = performance.now();

log.section("BUILD");
log.info(
  "Building @huggingface/transformers-react with TypeScript in watch mode...",
);

// Start TypeScript compiler in watch mode
const tsc = spawn("tsc", ["--watch", "--preserveWatchOutput"], {
  cwd: ROOT_DIR,
  stdio: "inherit",
  shell: true,
});

// Wait a bit for initial build to complete
await new Promise((resolve) => setTimeout(resolve, 3000));

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

// List all files in OUT_DIR (excluding .map files and .tsbuildinfo)
const files = readdirSync(OUT_DIR)
  .filter((file) => !file.endsWith(".map") && !file.endsWith(".tsbuildinfo"))
  .sort();

if (files.length > 0) {
  console.log(`${colors.bright}Available files:${colors.reset}`);
  files.forEach((file) => {
    log.url(`http://localhost:${PORT}/${file}`);
  });
}

console.log(
  `\n${colors.yellow}[watch]${colors.reset} Watching for changes...\n`,
);

// Keep process alive and cleanup
process.on("SIGINT", async () => {
  console.log(
    `\n\n${colors.yellow}[stop]${colors.reset} Stopping watch mode and server...`,
  );
  server.close();
  tsc.kill();
  log.dim("Goodbye!");
  process.exit(0);
});
