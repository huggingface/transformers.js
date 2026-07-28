import { build } from "esbuild";
import { mkdir } from "node:fs/promises";

await mkdir(new URL("../dist/", import.meta.url), { recursive: true });

const shared = {
  entryPoints: [new URL("../src/index.ts", import.meta.url).pathname],
  bundle: true,
  sourcemap: false,
  logLevel: "warning",
};

await Promise.all([
  build({
    ...shared,
    outfile: new URL("../dist/transformers-onnx.node.mjs", import.meta.url).pathname,
    platform: "node",
    format: "esm",
    external: ["onnxruntime-common", "onnxruntime-node"],
    alias: { "onnxruntime-web/webgpu": "./src/empty.ts" },
  }),
  build({
    ...shared,
    outfile: new URL("../dist/transformers-onnx.node.cjs", import.meta.url).pathname,
    platform: "node",
    format: "cjs",
    external: ["onnxruntime-common", "onnxruntime-node"],
    alias: { "onnxruntime-web/webgpu": "./src/empty.ts" },
  }),
  build({
    ...shared,
    outfile: new URL("../dist/transformers-onnx.web.js", import.meta.url).pathname,
    platform: "browser",
    format: "esm",
    external: ["onnxruntime-common", "onnxruntime-web"],
    alias: { "onnxruntime-node": "./src/empty.ts" },
  }),
  build({
    entryPoints: [new URL("../src/testing.ts", import.meta.url).pathname],
    bundle: true,
    sourcemap: false,
    logLevel: "warning",
    outfile: new URL("../dist/testing.mjs", import.meta.url).pathname,
    platform: "node",
    format: "esm",
    external: ["onnxruntime-common", "onnxruntime-node"],
  }),
  build({
    entryPoints: [new URL("../src/testing.ts", import.meta.url).pathname],
    bundle: true,
    sourcemap: false,
    logLevel: "warning",
    outfile: new URL("../dist/testing.cjs", import.meta.url).pathname,
    platform: "node",
    format: "cjs",
    external: ["onnxruntime-common", "onnxruntime-node"],
  }),
]);
