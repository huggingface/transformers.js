import { build as esbuild } from "esbuild";
import path from "node:path";
import { spawn } from "node:child_process";
import { OUT_DIR, getEsbuildProdConfig } from "./constants.mjs";
import { reportSize } from "../../../../../scripts/reportSize.mjs";
import { colors } from "../../../../../scripts/logger.mjs";
import { BUILD_TARGETS } from "./targets.mjs";

/**
 * Build a single target
 */
async function buildTarget({ outputFile, platform, minify }, log) {
  const config = getEsbuildProdConfig();

  log.build(`Building ${colors.bright}${outputFile}${colors.reset}...`);

  await esbuild({
    ...config,
    outfile: path.join(OUT_DIR, outputFile),
    platform,
    minify,
  });

  reportSize(path.join(OUT_DIR, outputFile), log);
}

/**
 * Generate TypeScript declarations
 */
async function generateTypeScriptDeclarations(rootDir, log) {
  log.build("Generating TypeScript declarations...");

  await new Promise((resolve, reject) => {
    const tsc = spawn("tsc", ["--declaration", "--emitDeclarationOnly"], {
      cwd: rootDir,
      stdio: "pipe",
      shell: true,
    });

    tsc.stdout.on("data", (data) => {
      const output = data.toString().trim();
      if (output) {
        output.split("\n").forEach((line) => {
          log.dim(line);
        });
      }
    });

    tsc.stderr.on("data", (data) => {
      const output = data.toString().trim();
      if (output) {
        output.split("\n").forEach((line) => {
          log.error(line);
        });
      }
    });

    tsc.on("exit", (code) => {
      if (code === 0) resolve();
      else
        reject(
          new Error(
            `TypeScript declaration generation failed with code ${code}`,
          ),
        );
    });
  });

  log.done("TypeScript declarations generated");
}

/**
 * Build all targets for production
 */
export async function buildAll(rootDir, log) {
  // Build all targets
  for (const target of BUILD_TARGETS) {
    log.section(target.name);
    await buildTarget(target.config, log);
  }

  // Generate TypeScript declarations after all builds
  log.section("TypeScript Declarations");
  await generateTypeScriptDeclarations(rootDir, log);
}
