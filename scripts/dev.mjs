#!/usr/bin/env node
import { spawn } from "node:child_process";
import { colors, log } from "./logger.mjs";

const processes = [];

// Cleanup function
const cleanup = () => {
  console.log(`\n\n${colors.yellow}[stop]${colors.reset} Stopping all dev servers...`);
  processes.forEach((proc) => {
    try {
      proc.kill("SIGINT");
    } catch (error) {
      // Ignore errors during cleanup
    }
  });
  process.exit(0);
};

// Handle various termination signals
process.on("SIGINT", cleanup);
process.on("SIGTERM", cleanup);
process.on("exit", cleanup);

log.section("DEV SERVERS");
log.info("Starting development servers for all packages...\n");

// Start transformers dev server
const transformers = spawn("npm", ["run", "dev", "--workspace=packages/transformers"], {
  stdio: "inherit",
  shell: true,
});
processes.push(transformers);

// Start transformers-react dev server
const transformersReact = spawn("npm", ["run", "dev", "--workspace=packages/transformers-react"], {
  stdio: "inherit",
  shell: true,
});
processes.push(transformersReact);

// Handle process exits
transformers.on("exit", (code) => {
  if (code !== 0 && code !== null) {
    log.error(`Transformers dev server exited with code ${code}`);
    cleanup();
  }
});

transformersReact.on("exit", (code) => {
  if (code !== 0 && code !== null) {
    log.error(`Transformers-React dev server exited with code ${code}`);
    cleanup();
  }
});

// Keep process alive
process.stdin.resume();
