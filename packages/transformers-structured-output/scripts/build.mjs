import { build } from "esbuild";
import { rmSync } from "node:fs";

rmSync("dist", { recursive: true, force: true });
rmSync("types", { recursive: true, force: true });

const common = {
  entryPoints: ["src/index.ts"],
  bundle: true,
  platform: "neutral",
  target: "es2022",
  sourcemap: true,
  external: ["@huggingface/transformers"],
};

await Promise.all([
  build({
    ...common,
    format: "esm",
    outfile: "dist/index.js",
  }),
  build({
    ...common,
    format: "cjs",
    outfile: "dist/index.cjs",
  }),
]);
