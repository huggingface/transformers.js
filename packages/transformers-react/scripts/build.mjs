import { build } from "esbuild";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import { colors, createLogger } from "../../../scripts/logger.mjs";
import prepareOutDir from "../../../scripts/prepareOutDir.mjs";
import { reportSize } from "../../../scripts/reportSize.mjs";

const log = createLogger("react");
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.join(__dirname, "..");
const OUT_DIR = path.join(ROOT_DIR, "dist");
const SRC_DIR = path.join(ROOT_DIR, "src");

const startTime = performance.now();

log.section("BUILD");
log.info("Building @huggingface/transformers-react...\n");

prepareOutDir(OUT_DIR);

try {
  // Build 1: ESM bundle for npm (bundled with external deps)
  log.build("Building ESM for npm...");
  await build({
    entryPoints: [path.join(SRC_DIR, "index.ts")],
    bundle: true,
    outfile: path.join(OUT_DIR, "index.js"),
    format: "esm",
    platform: "neutral",
    target: "es2020",
    sourcemap: true,
    external: ["react", "@huggingface/transformers"],
    minify: false,
    logLevel: "silent",
  });
  log.done("ESM build complete");

  // Build 2: TypeScript declarations
  log.build("Generating TypeScript declarations...");
  await new Promise((resolve, reject) => {
    const tsc = spawn("tsc", ["--declaration", "--emitDeclarationOnly"], {
      cwd: ROOT_DIR,
      stdio: "pipe",
      shell: true,
    });

    // Capture and prefix output
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

  // Build 3: Browser bundle with URL imports
  log.build("Building browser bundle...");
  await build({
    entryPoints: [path.join(SRC_DIR, "index.ts")],
    bundle: true,
    outfile: path.join(OUT_DIR, "browser.js"),
    format: "esm",
    platform: "browser",
    target: "es2020",
    sourcemap: true,
    external: ["react", "@huggingface/transformers"],
    minify: false,
    logLevel: "silent",
  });
  reportSize(path.join(OUT_DIR, "browser.js"), log);

  // Build 4: Minified browser bundle
  log.build("Building minified browser bundle...");
  await build({
    entryPoints: [path.join(SRC_DIR, "index.ts")],
    bundle: true,
    outfile: path.join(OUT_DIR, "browser.min.js"),
    format: "esm",
    platform: "browser",
    target: "es2020",
    sourcemap: true,
    external: ["react", "@huggingface/transformers"],
    minify: true,
    logLevel: "silent",
  });
  reportSize(path.join(OUT_DIR, "browser.min.js"), log);

  const endTime = performance.now();
  const duration = (endTime - startTime).toFixed(2);

  log.section("SUMMARY");
  log.success(
    `Build completed in ${colors.bright}${duration}ms${colors.reset}\n`,
  );

  log.info(`${colors.bright}Output files:${colors.reset}`);
  log.file(
    `${colors.cyan}dist/index.js${colors.reset} - ESM for npm (external deps)`,
  );
  log.file(
    `${colors.cyan}dist/index.d.ts${colors.reset} - TypeScript declarations`,
  );
  log.file(`${colors.cyan}dist/browser.js${colors.reset} - Browser bundle`);
  log.file(
    `${colors.cyan}dist/browser.min.js${colors.reset} - Minified browser bundle`,
  );
} catch (error) {
  log.error(`Build failed: ${error.message}`);
  console.error(error);
  process.exit(1);
}
