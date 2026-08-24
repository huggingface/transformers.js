import { execFile } from "node:child_process";
import { mkdtemp, mkdir, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";

const exec = promisify(execFile);
const root = new URL("..", import.meta.url).pathname;
const temporaryDirectory = await mkdtemp(join(tmpdir(), "transformers-js-packages-"));
const packDirectory = join(temporaryDirectory, "packages");
const consumerDirectory = join(temporaryDirectory, "consumer");

try {
  await mkdir(packDirectory);
  await mkdir(consumerDirectory);

  for (const packageName of ["@huggingface/transformers-onnxruntime", "@huggingface/transformers"]) {
    await exec("pnpm", ["--filter", packageName, "pack", "--pack-destination", packDirectory], { cwd: root });
  }

  const tarballs = (await readdir(packDirectory)).map((file) => join(packDirectory, file));
  await writeFile(join(consumerDirectory, "package.json"), '{"private":true,"type":"module"}\n');

  const env = { ...process.env };
  delete env.ONNXRUNTIME_NODE_INSTALL;
  await exec("npm", ["install", "--no-package-lock", ...tarballs], { cwd: consumerDirectory, env });
  await exec(
    process.execPath,
    [
      "--input-type=module",
      "--eval",
      "await import('@huggingface/transformers'); await import('@huggingface/transformers-onnxruntime'); await import('onnxruntime-node');",
    ],
    { cwd: consumerDirectory },
  );
} finally {
  await rm(temporaryDirectory, { recursive: true, force: true });
}
