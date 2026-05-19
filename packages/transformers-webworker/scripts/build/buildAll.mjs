import { build as esbuild } from "esbuild";
import path from "node:path";
import { OUT_DIR, ROOT_DIR, getEsbuildProdConfig } from "./constants.mjs";
import { reportSize } from "../../../../scripts/reportSize.mjs";
import { colors } from "../../../../scripts/logger.mjs";
import { BUILD_TARGETS } from "./targets.mjs";

/**
 * Helper function to create build configurations.
 */
async function buildTarget({ name = "", suffix = ".js", format = "esm", externalModules = [] }, log) {
  const regularFile = `index${name}${suffix}`;
  const minFile = `index${name}.min${suffix}`;

  log.build(`Building ${colors.bright}${regularFile}${colors.reset}...`);
  await esbuild({
    ...getEsbuildProdConfig(ROOT_DIR),
    format,
    outfile: path.join(OUT_DIR, regularFile),
    external: externalModules,
  });
  reportSize(path.join(OUT_DIR, regularFile), log);

  log.build(`Building ${colors.bright}${minFile}${colors.reset}...`);
  await esbuild({
    ...getEsbuildProdConfig(ROOT_DIR),
    format,
    outfile: path.join(OUT_DIR, minFile),
    minify: true,
    external: externalModules,
    legalComments: "none",
  });
  reportSize(path.join(OUT_DIR, minFile), log);
}

/**
 * Build all targets for production
 */
export async function buildAll(log) {
  for (const target of BUILD_TARGETS) {
    log.section(target.name);
    await buildTarget(target.config, log);
  }
}
