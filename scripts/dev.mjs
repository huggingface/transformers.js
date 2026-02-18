#!/usr/bin/env node

/**
 * This scripts looks at all packages and starts their dev command in parallel, if they have one.
 */

import { spawn } from "node:child_process";
import { readdirSync, readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
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

// Find all packages with dev scripts
const findPackagesWithDevScript = () => {
  const packagesDir = "packages";
  const packagesWithDev = [];

  if (!existsSync(packagesDir)) {
    log.error(`Packages directory not found: ${packagesDir}`);
    return packagesWithDev;
  }

  const packages = readdirSync(packagesDir, { withFileTypes: true })
    .filter((dirent) => dirent.isDirectory())
    .map((dirent) => dirent.name);

  for (const pkg of packages) {
    const packageJsonPath = join(packagesDir, pkg, "package.json");

    if (existsSync(packageJsonPath)) {
      try {
        const packageJson = JSON.parse(readFileSync(packageJsonPath, "utf-8"));
        if (packageJson.scripts && packageJson.scripts.dev) {
          packagesWithDev.push({
            name: pkg,
            displayName: packageJson.name || pkg,
          });
        }
      } catch (error) {
        log.error(`Failed to read package.json for ${pkg}: ${error.message}`);
      }
    }
  }

  return packagesWithDev;
};

log.section("DEV SERVERS");
log.info("Discovering packages with dev scripts...\n");

const packagesWithDev = findPackagesWithDevScript();

if (packagesWithDev.length === 0) {
  log.error("No packages found with dev scripts!");
  process.exit(1);
}

log.info(`Found ${packagesWithDev.length} package(s) with dev scripts:`);
packagesWithDev.forEach((pkg) => {
  log.info(`  - ${pkg.displayName} (${pkg.name})`);
});
log.info("\nStarting development servers...\n");

// Start dev server for each package
packagesWithDev.forEach((pkg) => {
  const proc = spawn("pnpm", ["--filter", pkg.displayName, "dev"], {
    stdio: "inherit",
    shell: true,
  });

  processes.push(proc);

  // Handle process exits
  proc.on("exit", (code) => {
    if (code !== 0 && code !== null) {
      log.error(`${pkg.displayName} dev server exited with code ${code}`);
      cleanup();
    }
  });
});

// Keep process alive
process.stdin.resume();
