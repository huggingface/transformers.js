import { context } from "esbuild";
import { rmSync } from "node:fs";

rmSync("dist", { recursive: true, force: true });

const common = {
  entryPoints: ["src/index.ts"],
  bundle: true,
  platform: "neutral",
  target: "es2022",
  sourcemap: true,
  external: ["@huggingface/transformers", "llguidance"],
};

const contexts = await Promise.all([
  context({
    ...common,
    format: "esm",
    outfile: "dist/index.js",
  }),
  context({
    ...common,
    format: "cjs",
    outfile: "dist/index.cjs",
  }),
]);

await Promise.all(contexts.map((ctx) => ctx.watch()));
console.log("Watching @huggingface/transformers-llguidance...");

process.on("SIGINT", async () => {
  await Promise.all(contexts.map((ctx) => ctx.dispose()));
  process.exit(0);
});
