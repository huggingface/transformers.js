import path from "node:path";
import { fileURLToPath } from "node:url";
import { colors, createLogger } from "../../../scripts/logger.mjs";
import prepareOutDir from "../../../scripts/prepareOutDir.mjs";
import { OUT_DIR } from "./esbuild/build/constants.mjs";
import { buildAll } from "./esbuild/build/buildAll.mjs";

const log = createLogger("react");
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT_DIR = path.join(__dirname, "..");

const startTime = performance.now();

log.section("BUILD");
log.info("Building @huggingface/transformers-react...\n");

prepareOutDir(OUT_DIR);

try {
  await buildAll(ROOT_DIR, log);

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
