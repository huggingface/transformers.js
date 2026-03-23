import { build as esbuild } from "esbuild";
import path from "node:path";
import { stripNodePrefixPlugin } from "./plugins/stripNodePrefixPlugin.mjs";
import { ignoreModulesPlugin } from "./plugins/ignoreModulesPlugin.mjs";
import { postBuildPlugin } from "./plugins/postBuildPlugin.mjs";
import { externalNodeBuiltinsPlugin } from "./plugins/externalNodeBuiltinsPlugin.mjs";
import { OUT_DIR, ROOT_DIR, getEsbuildProdConfig } from "./constants.mjs";
import { reportSize } from "../../../../scripts/reportSize.mjs";
import { colors } from "../../../../scripts/logger.mjs";
import { BUILD_TARGETS } from "./targets.mjs";

/**
 * Helper function to create build configurations.
 */
async function buildTarget(
  {
    name = "",
    suffix = ".js",
    format = "esm", // 'esm' | 'cjs'
    ignoreModules = [],
    externalModules = [],
    usePostBuild = false,
  },
  log,
) {
  const platform = format === "cjs" ? "node" : "neutral";

  const regularFile = `transformers${name}${suffix}`;
  const minFile = `transformers${name}.min${suffix}`;

  const plugins = [];
  // Add ignoreModulesPlugin FIRST so it can catch modules before stripNodePrefixPlugin marks them as external
  if (ignoreModules.length > 0) {
    plugins.push(ignoreModulesPlugin(ignoreModules));
  }
  plugins.push(stripNodePrefixPlugin());
  plugins.push(externalNodeBuiltinsPlugin());
  if (usePostBuild) {
    plugins.push(postBuildPlugin(OUT_DIR, ROOT_DIR));
  }

  log.build(`Building ${colors.bright}${regularFile}${colors.reset}...`);
  await esbuild({
    ...getEsbuildProdConfig(ROOT_DIR),
    platform,
    format,
    outfile: path.join(OUT_DIR, regularFile),
    external: externalModules,
    plugins,
  });
  reportSize(path.join(OUT_DIR, regularFile), log);

  log.build(`Building ${colors.bright}${minFile}${colors.reset}...`);
  await esbuild({
    ...getEsbuildProdConfig(ROOT_DIR),
    platform,
    format,
    outfile: path.join(OUT_DIR, minFile),
    minify: true,
    external: externalModules,
    plugins,
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
