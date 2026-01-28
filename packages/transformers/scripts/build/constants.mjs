import path from "node:path";
import { fileURLToPath } from "node:url";

export const DIST_FOLDER = "dist";
export const NODE_IGNORE_MODULES = ["onnxruntime-web"];
export const NODE_EXTERNAL_MODULES = [
  "onnxruntime-common",
  "onnxruntime-node",
  "sharp",
  // node:* modules are handled by externalNodeBuiltinsPlugin
];

export const WEB_IGNORE_MODULES = ["onnxruntime-node", "sharp", "fs", "path", "url", "stream", "stream/promises"];
export const WEB_EXTERNAL_MODULES = ["onnxruntime-common", "onnxruntime-web"];

const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const ROOT_DIR = path.join(__dirname, "../..");
export const OUT_DIR = path.join(ROOT_DIR, DIST_FOLDER);

export const getEsbuildDevConfig = (rootDir) => ({
  bundle: true,
  treeShaking: true,
  logLevel: "info",
  entryPoints: [path.join(rootDir, "src/transformers.js")],
  platform: "neutral",
  format: "esm",
  sourcemap: true,
  logOverride: {
    // Suppress import.meta warning for CJS builds - it's handled gracefully in the code
    "empty-import-meta": "silent",
  },
});

export const getEsbuildProdConfig = (rootDir) => ({
  ...getEsbuildDevConfig(rootDir),
  logLevel: "warning",
  sourcemap: false,
});
