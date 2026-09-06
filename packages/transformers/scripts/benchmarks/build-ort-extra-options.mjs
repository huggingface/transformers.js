// Build the runtime prerequisite for benchmark/review until ORT #32456 is released.
import { build } from "esbuild";
import { createRequire } from "node:module";
import { parseArgs } from "node:util";
import { readFile } from "node:fs/promises";
import path from "node:path";
const { values } = parseArgs({ options: { output: { type: "string", default: ".cache/ort.webgpu.extra-options.mjs" } } });
const require = createRequire(import.meta.url);
const pkg = path.dirname(path.dirname(require.resolve("onnxruntime-web")));
await build({
  entryPoints: [path.join(pkg, "lib/index.ts")],
  outfile: path.resolve(values.output),
  bundle: true,
  format: "esm",
  platform: "browser",
  minify: true,
  external: ["node:*", "fs", "path", "os", "url", "module", "worker_threads"],
  define: {
    "BUILD_DEFS.IS_ESM": "true",
    "BUILD_DEFS.ENABLE_BUNDLE_WASM_JS": "false",
    "BUILD_DEFS.ENABLE_JSPI": "false",
    "BUILD_DEFS.DISABLE_JSEP": "true",
    "BUILD_DEFS.DISABLE_WEBGPU": "false",
    "BUILD_DEFS.DISABLE_WASM": "false",
    "BUILD_DEFS.DISABLE_WEBGL": "true",
    "BUILD_DEFS.DISABLE_WEBNN": "true",
    "BUILD_DEFS.DISABLE_WASM_PROXY": "true",
    "BUILD_DEFS.BUNDLE_FILENAME": '"ort.webgpu.extra-options.mjs"',
    "BUILD_DEFS.ESM_IMPORT_META_URL": "import.meta.url",
  },
  plugins: [
    {
      name: "extra-before-ep",
      setup(build) {
        build.onLoad({ filter: /wasm[\\/]session-options\.ts$/ }, async ({ path }) => {
          let source = (await readFile(path, "utf8")).replace(/\r\n/g, "\n");
          const start = source.indexOf("    if (sessionOptions.extra !== undefined) {");
          const end = source.indexOf("    return [sessionOptionsHandle, allocs];", start);
          if (start < 0 || end < 0) throw new Error("Missing patch target");
          const provider = source.indexOf("    if (sessionOptions.executionProviders) {");
          if (provider < 0) throw new Error("Missing execution-provider setup");
          if (start < provider) return { contents: source, loader: "ts" };
          const block = source.slice(start, end);
          source = source.slice(0, start) + source.slice(end);
          const insert = source.indexOf("    if (sessionOptions.executionProviders) {");
          source = source.slice(0, insert) + block + source.slice(insert);
          return { contents: source, loader: "ts" };
        });
      },
    },
  ],
});
console.log("Built pinned ORT Web with extra settings applied before EP construction.");
