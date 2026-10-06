import { build } from "esbuild";
import { mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";

await mkdir(new URL("../dist/", import.meta.url), { recursive: true });

await build({
  entryPoints: [fileURLToPath(new URL("../src/index.ts", import.meta.url))],
  bundle: true,
  sourcemap: false,
  logLevel: "warning",
  outfile: fileURLToPath(new URL("../dist/transformers-litertjs-runtime.js", import.meta.url)),
  platform: "browser",
  format: "esm",
  external: ["@huggingface/transformers", "@litertjs/core"],
});
