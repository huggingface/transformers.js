import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawn } from "node:child_process";
import process from "node:process";
import { build } from "esbuild";

const tempDir = await mkdtemp(join(tmpdir(), "transformers-agent-tests-"));
const outfile = join(tempDir, "tests.mjs");

try {
  await build({
    entryPoints: ["tests/Gemma4ParserStrategy.test.ts"],
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
