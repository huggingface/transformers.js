import TerserPlugin from "terser-webpack-plugin";
import { fileURLToPath } from "url";
import path from "path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/**
 * Helper function to create webpack configurations.
 * @param {Object} options Options for creating a webpack target.
 * @param {string} options.name Name of output file.
 * @param {string} options.suffix Suffix of output file.
 * @param {string} options.type Type of library.
 * @param {string} options.ignoreModules The list of modules to ignore.
 * @param {string} options.externalModules The list of modules to set as external.
 * @returns {import('webpack').Configuration} One webpack target.
 */
function buildConfig({
  name = "",
  suffix = ".js",
  type = "module", // 'module' | 'commonjs'
  ignoreModules = [],
  externalModules = [],
} = {}) {
  const outputModule = type === "module";

  const alias = Object.fromEntries(
    ignoreModules.map((module) => [module, false]),
  );

  /** @type {import('webpack').Configuration} */
  const config = {
    mode: "development",
    devtool: "source-map",
    entry: {
      [`transformers${name}`]: "./src/transformers.js",
      [`transformers${name}.min`]: "./src/transformers.js",
    },
    output: {
      filename: `[name]${suffix}`,
      path: path.join(__dirname, "dist"),
      library: {
        type,
      },
      assetModuleFilename: "[name][ext]",
      chunkFormat: "module",
    },
    optimization: {
      minimize: true,
      minimizer: [
        new TerserPlugin({
          test: new RegExp(`\\.min\\${suffix}$`),

          // Do not bundle with comments.
          // See https://webpack.js.org/plugins/terser-webpack-plugin/#remove-comments for more information.
          terserOptions: {
            output: {
              comments: false,
            },
          },
          extractComments: false,
        }),
      ],
    },
    experiments: {
      outputModule,
    },
    resolve: { alias },

    externals: externalModules,

    // Development server
    devServer: {
      static: {
        directory: __dirname,
      },
      port: 8080,
    },
  };

  if (outputModule) {
    config.module = {
      parser: {
        javascript: {
          importMeta: false,
        },
      },
    };
  } else {
    config.externalsType = "commonjs";
  }

  return config;
}

// Do not bundle onnxruntime-web when packaging for Node.js.
// Instead, we use the native library (onnxruntime-node).
const NODE_IGNORE_MODULES = ["onnxruntime-web"];

// Do not bundle the following modules with webpack (mark as external)
// NOTE: This is necessary for both type="module" and type="commonjs",
// and will be ignored when building for web (only used for node/deno)
const NODE_EXTERNAL_MODULES = [
  "onnxruntime-node",
  "sharp",
  "fs",
  "path",
  "url",
];

// Web-only build
const WEB_BUILD = buildConfig({
  type: "module",
});

// Node-compatible builds
const NODE_BUILDS = [
  buildConfig({
    suffix: ".mjs",
    type: "module",
    ignoreModules: NODE_IGNORE_MODULES,
    externalModules: NODE_EXTERNAL_MODULES,
  }),
  buildConfig({
    suffix: ".cjs",
    type: "commonjs",
    ignoreModules: NODE_IGNORE_MODULES,
    externalModules: NODE_EXTERNAL_MODULES,
  }),
];

// When running with `webpack serve`, only build the web target.
const BUILDS = process.env.WEBPACK_SERVE
  ? [WEB_BUILD]
  : [WEB_BUILD, ...NODE_BUILDS];
export default BUILDS;
