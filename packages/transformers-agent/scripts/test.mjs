import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { spawn } from "node:child_process";
import process from "node:process";
import { build } from "esbuild";

const tempDir = await mkdtemp(join(tmpdir(), "transformers-agent-tests-"));
const outfile = join(tempDir, "tests.mjs");
const entryPoint = join(tempDir, "entry.mjs");
const transformersStub = join(tempDir, "transformers-stub.mjs");

try {
  await writeFile(
    transformersStub,
    [
      "export class DynamicCache {}",
      "export class TextStreamer { constructor(_tokenizer, options) { Object.assign(this, options); } }",
      "export class AutoTokenizer { static calls = []; static async from_pretrained(...args) { this.calls.push(args); return {}; } }",
      "export class AutoModelForCausalLM { static calls = []; static async from_pretrained(...args) { this.calls.push(args); return {}; } }",
      "export class ModelRegistry {",
      "  static calls = [];",
      "  static async is_pipeline_cached(...args) { this.calls.push(['is_pipeline_cached', ...args]); return true; }",
      "  static async get_pipeline_files(...args) { this.calls.push(['get_pipeline_files', ...args]); return ['model.onnx']; }",
      "  static async get_file_metadata(...args) { this.calls.push(['get_file_metadata', ...args]); return { size: 10, fromCache: true }; }",
      "}",
    ].join("\n"),
  );
  await writeFile(
    entryPoint,
    [resolve("tests/Agent.test.ts"), resolve("tests/Model.test.ts"), resolve("tests/MessageFormatting.test.ts")]
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
    alias: { "@huggingface/transformers": transformersStub },
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
