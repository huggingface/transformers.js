import { build as esbuild } from "esbuild";
import path from "node:path";
import { stripNodePrefixPlugin } from "./build/plugins/stripNodePrefixPlugin.mjs";
import { ignoreModulesPlugin } from "./build/plugins/ignoreModulesPlugin.mjs";
import { postBuildPlugin } from "./build/plugins/postBuildPlugin.mjs";
import { externalNodeBuiltinsPlugin } from "./build/plugins/externalNodeBuiltinsPlugin.mjs";
import {
  NODE_IGNORE_MODULES,
  NODE_EXTERNAL_MODULES,
  WEB_IGNORE_MODULES,
  WEB_EXTERNAL_MODULES,
  OUT_DIR,
  ROOT_DIR,
  getEsbuildProdConfig,
} from "./build/constants.mjs";
import { reportSize } from "./build/reportSize.mjs";
import prepareOutDir from "./build/prepareOutDir.mjs";

/**
 *
 * Helper function to create build configurations.
 * Equivalent to webpack's buildConfig function.
 */
async function buildTarget({
  name = "",
  suffix = ".js",
  format = "esm", // 'esm' | 'cjs'
  ignoreModules = [],
  externalModules = [],
  usePostBuild = false,
}) {
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

  console.log(`\nBuilding ${regularFile}...`);
  await esbuild({
    ...getEsbuildProdConfig(ROOT_DIR),
    platform,
    format,
    outfile: path.join(OUT_DIR, regularFile),
    external: externalModules,
    plugins,
  });
  reportSize(path.join(OUT_DIR, regularFile));

  console.log(`\nBuilding ${minFile}...`);
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
  reportSize(path.join(OUT_DIR, minFile));
}

console.log("\nBuilding transformers.js with esbuild...\n");

const startTime = performance.now();

try {
  prepareOutDir(OUT_DIR);

  // Bundle build - bundles everything except ignored modules
  console.log("\n=== Bundle Build (ESM) ===");
  await buildTarget({
    name: "",
    suffix: ".js",
    format: "esm",
    ignoreModules: WEB_IGNORE_MODULES,
    externalModules: [],
    usePostBuild: true,
  });

  // Web build - external onnxruntime libs
  console.log("\n=== Web Build (ESM) ===");
  await buildTarget({
    name: ".web",
    suffix: ".js",
    format: "esm",
    ignoreModules: WEB_IGNORE_MODULES,
    externalModules: WEB_EXTERNAL_MODULES,
    usePostBuild: false,
  });

  // Node ESM build
  console.log("\n=== Node Build (ESM) ===");
  await buildTarget({
    name: ".node",
    suffix: ".mjs",
    format: "esm",
    ignoreModules: NODE_IGNORE_MODULES,
    externalModules: NODE_EXTERNAL_MODULES,
    usePostBuild: false,
  });

  // Node CJS build
  console.log("\n=== Node Build (CJS) ===");
  await buildTarget({
    name: ".node",
    suffix: ".cjs",
    format: "cjs",
    ignoreModules: NODE_IGNORE_MODULES,
    externalModules: NODE_EXTERNAL_MODULES,
    usePostBuild: false,
  });

  // Subpath exports - React
  console.log("\n=== Subpath: /react (ESM) ===");
  const reactEntry = path.join(ROOT_DIR, "src/react/index.ts");
  if (await import("node:fs").then((fs) => fs.promises.access(reactEntry).then(() => true).catch(() => false))) {
    await esbuild({
      entryPoints: [reactEntry],
      bundle: true,
      format: "esm",
      platform: "neutral",
      outfile: path.join(OUT_DIR, "react.js"),
      external: ["react", "@huggingface/transformers"],
      sourcemap: false,
      logLevel: "warning",
      treeShaking: true,
    });
    reportSize(path.join(OUT_DIR, "react.js"));
  } else {
    console.log("Skipping /react build (entry point not found)");
  }

  const endTime = performance.now();
  const duration = (endTime - startTime).toFixed(2);
  console.log(`\nAll builds completed successfully in ${duration}ms!\n`);
} catch (error) {
  console.error("\nBuild failed:", error);
  process.exit(1);
}
