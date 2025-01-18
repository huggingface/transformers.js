import TerserPlugin from "terser-webpack-plugin";
import { fileURLToPath } from "url";
import path from "path";
import fs from "fs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/**
 * Plugin to post-process build files. Required to solve certain issues with ESM module output.
 * See https://github.com/webpack/webpack/issues/17121 for more information.
 * 
 * @see https://webpack.js.org/contribute/writing-a-plugin/
 */
class PostBuildPlugin {

  apply(compiler) {
    compiler.hooks.done.tap('PostBuildPlugin', () => {
      const dist = path.join(__dirname, 'dist');
      const ORT_JSEP_FILE = 'ort-wasm-simd-threaded.jsep.mjs';
      const ORT_BUNDLE_FILE = 'ort.bundle.min.mjs';

      // 1. Remove unnecessary files
      {
        const file = path.join(dist, ORT_BUNDLE_FILE);
        if (fs.existsSync(file)) fs.unlinkSync(file);
      }
      
      // 2. Copy unbundled JSEP file
      {
        const src = path.join(__dirname, 'node_modules/onnxruntime-web/dist', ORT_JSEP_FILE);
        const dest = path.join(dist, ORT_JSEP_FILE);
        fs.copyFileSync(src, dest);
      }

      // 3. Replace strings in certain files
      {
        const files = ['transformers.js', 'transformers.min.js'];
        for (const file of files) {
          const filePath = path.join(dist, file);
          let content = fs.readFileSync(filePath, 'utf8');
          content = content.replace(
            // Replace all instances of `new URL("./", import.meta.url)` with `new URL(import.meta.url)`,
            // as it causes several issues with build tools and bundlers.
            // 
            // See the following issues for more information:
            // - https://github.com/huggingface/transformers.js/issues/911
            // - https://github.com/huggingface/transformers.js/issues/984
            // - https://github.com/huggingface/transformers.js/issues/980
            // - https://github.com/huggingface/transformers.js/issues/1021
            // - https://github.com/huggingface/transformers.js/issues/1026
            new RegExp('new URL\\(["\']\\.\\\/["\'],\\s*import\\.meta\\.url\\)', 'gm'),
            "new URL(import.meta.url)",
          );
          fs.writeFileSync(filePath, content, 'utf8');
        }
      }
    });
  }
}

/**
 * Helper function to create webpack configurations.
 * @param {Object} options Options for creating a webpack target.
 * @param {string} options.name Name of output file.
 * @param {string} options.suffix Suffix of output file.
 * @param {string} options.type Type of library.
 * @param {string} options.ignoreModules The list of modules to ignore.
 * @param {string} options.externalModules The list of modules to set as external.
 * @param {Object[]} options.plugins List of plugins to use.
 * @returns {import('webpack').Configuration} One webpack target.
 */
function buildConfig({
  name = "",
  suffix = ".js",
  type = "module", // 'module' | 'commonjs'
  ignoreModules = [],
  externalModules = [],
  plugins = [],
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
    plugins,
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
  plugins: [new PostBuildPlugin()],
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
