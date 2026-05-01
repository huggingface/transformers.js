import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { spawn } from "node:child_process";
import process from "node:process";
import { build } from "esbuild";

const tempDir = await mkdtemp(join(tmpdir(), "transformers-agent-tests-"));
const outfile = join(tempDir, "tests.mjs");
const entryPoint = join(tempDir, "entry.mjs");

try {
  await writeFile(
    entryPoint,
    [resolve("tests/GraniteParserStrategy.test.ts"), resolve("tests/Gemma4ParserStrategy.test.ts"), resolve("tests/Qwen3ParserStrategy.test.ts")]
      .map((path) => `import ${JSON.stringify(path)};`)
      .join("\n"),
  );

  await build({
    entryPoints: [entryPoint],
    outfile,
    bundle: true,
    format: "esm",
    platform: "node",
    target: "node20",
    external: ["node:*"],
    logLevel: "silent",
  });

  const exitCode = await new Promise((resolve, reject) => {
    const child = spawn(process.execPath, ["--test", outfile], { stdio: "inherit" });
    child.once("exit", (code) => resolve(code ?? 1));
    child.once("error", reject);
  });

  if (exitCode !== 0) {
    process.exitCode = exitCode;
  }
} finally {
  await rm(tempDir, { recursive: true, force: true });
}
