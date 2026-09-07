import { context } from "esbuild";
import { spawn } from "node:child_process";
import { rmSync } from "node:fs";

rmSync("dist", { recursive: true, force: true });
rmSync("types", { recursive: true, force: true });

const watchLogger = {
  name: "watch-logger",
  setup(build) {
    let startTime = 0;

    build.onStart(() => {
      startTime = performance.now();
      console.log(`[transformers-structured-output] rebuilding ${build.initialOptions.outfile}...`);
    });

    build.onEnd((result) => {
      const duration = (performance.now() - startTime).toFixed(2);
      if (result.errors.length > 0) {
        console.log(`[transformers-structured-output] rebuild failed in ${duration}ms`);
      } else {
        console.log(`[transformers-structured-output] rebuilt ${build.initialOptions.outfile} in ${duration}ms`);
      }
    });
  },
};

const common = {
  entryPoints: ["src/index.ts"],
  bundle: true,
  platform: "neutral",
  target: "es2022",
  sourcemap: true,
  external: ["@huggingface/transformers"],
  plugins: [watchLogger],
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

const tscWatch = spawn("tsc", ["--build", "--watch", "--preserveWatchOutput"], {
  stdio: "inherit",
  shell: true,
});

console.log("Watching @huggingface/transformers-structured-output...");

process.on("SIGINT", async () => {
  tscWatch.kill();
  await Promise.all(contexts.map((ctx) => ctx.dispose()));
  process.exit(0);
});
