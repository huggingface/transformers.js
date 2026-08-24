import { build } from "esbuild";
import { copyFile, mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";

await mkdir(new URL("../dist/", import.meta.url), { recursive: true });

const filePath = (path) => fileURLToPath(new URL(path, import.meta.url));
const entry = filePath("../src/index.ts");
const testingEntry = filePath("../src/testing.ts");
const empty = filePath("../src/empty.ts");
const wasmAssets = ["ort-wasm-simd-threaded.asyncify.mjs", "ort-wasm-simd-threaded.asyncify.wasm", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"];

const shared = {
  entryPoints: [entry],
  bundle: true,
  sourcemap: false,
  logLevel: "warning",
};

await Promise.all([
  build({
    ...shared,
    outfile: filePath("../dist/transformers-onnxruntime.node.mjs"),
    platform: "node",
    format: "esm",
    external: ["onnxruntime-common", "onnxruntime-node"],
    alias: { "onnxruntime-web/webgpu": empty },
    banner: { js: "const __ONNX_MODULE_URL__ = import.meta.url;" },
  }),
  build({
    ...shared,
    outfile: filePath("../dist/transformers-onnxruntime.node.cjs"),
    platform: "node",
    format: "cjs",
    external: ["onnxruntime-common", "onnxruntime-node"],
    alias: { "onnxruntime-web/webgpu": empty },
    banner: { js: 'const __ONNX_MODULE_URL__ = require("node:url").pathToFileURL(__filename).href;' },
  }),
  build({
    ...shared,
    outfile: filePath("../dist/transformers-onnxruntime.web.js"),
    platform: "browser",
    format: "esm",
    external: ["onnxruntime-common", "onnxruntime-web"],
    alias: { "onnxruntime-node": empty },
    banner: { js: "const __ONNX_MODULE_URL__ = import.meta.url;" },
  }),
  build({
    entryPoints: [testingEntry],
    bundle: true,
    sourcemap: false,
    logLevel: "warning",
    outfile: filePath("../dist/testing.mjs"),
    platform: "node",
    format: "esm",
    external: ["onnxruntime-common", "onnxruntime-node"],
  }),
  build({
    entryPoints: [testingEntry],
    bundle: true,
    sourcemap: false,
    logLevel: "warning",
    outfile: filePath("../dist/testing.cjs"),
    platform: "node",
    format: "cjs",
    external: ["onnxruntime-common", "onnxruntime-node"],
  }),
  ...wasmAssets.map((file) => copyFile(filePath(`../node_modules/onnxruntime-web/dist/${file}`), filePath(`../dist/${file}`))),
]);
