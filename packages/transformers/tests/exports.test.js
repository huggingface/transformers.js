import { mkdirSync, mkdtempSync, readFileSync, readdirSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, relative } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { spawnSync } from "node:child_process";

import * as transformers from "../src/transformers.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PACKAGE_DIR = join(__dirname, "..");
const MODELS_DIR = join(__dirname, "..", "src", "models");

function findModelingFiles(dir) {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);

    if (entry.isDirectory()) {
      return findModelingFiles(path);
    }
    return entry.isFile() && entry.name.startsWith("modeling_") && entry.name.endsWith(".js") ? [path] : [];
  });
}

function isPublicModelingFile(file) {
  return !file.endsWith(join("models", "modeling_utils.js"));
}

describe("Public exports", () => {
  it("resolves the root entry point and explicit browser subpath to their intended bundles", () => {
    const fixtureDir = mkdtempSync(join(tmpdir(), "transformers-js-exports-"));
    const fixturePackageDir = join(fixtureDir, "node_modules", "@huggingface", "transformers");
    const fixtureDistDir = join(fixturePackageDir, "dist");

    try {
      mkdirSync(fixtureDistDir, { recursive: true });
      const resolvedFixturePackageDir = realpathSync(fixturePackageDir);
      writeFileSync(join(fixturePackageDir, "package.json"), readFileSync(join(PACKAGE_DIR, "package.json")));
      writeFileSync(join(fixtureDistDir, "transformers.node.cjs"), 'module.exports = { target: "node-cjs" };');
      writeFileSync(join(fixtureDistDir, "transformers.node.mjs"), 'export const target = "node-esm";');
      writeFileSync(join(fixtureDistDir, "transformers.web.js"), 'export const target = "web";');

      const cjs = spawnSync(process.execPath, ["-e", 'console.log(JSON.stringify([require.resolve("@huggingface/transformers"), require.resolve("@huggingface/transformers/dist/transformers.web.js")]))'], { cwd: fixtureDir, encoding: "utf8" });
      expect(cjs.stderr).toBe("");
      expect(cjs.status).toBe(0);
      expect(JSON.parse(cjs.stdout).map((path) => relative(resolvedFixturePackageDir, path))).toEqual(["dist/transformers.node.cjs", "dist/transformers.web.js"]);

      const esm = spawnSync(process.execPath, ["--input-type=module", "-e", 'const root = await import("@huggingface/transformers"); const web = await import("@huggingface/transformers/dist/transformers.web.js"); console.log(JSON.stringify([root.target, web.target]));'], { cwd: fixtureDir, encoding: "utf8" });
      expect(esm.stderr).toBe("");
      expect(esm.status).toBe(0);
      expect(JSON.parse(esm.stdout)).toEqual(["node-esm", "web"]);
    } finally {
      rmSync(fixtureDir, { recursive: true, force: true });
    }
  });

  it("exports every public modeling_* symbol from the root entry point", async () => {
    const missing = [];

    for (const file of findModelingFiles(MODELS_DIR).filter(isPublicModelingFile).sort()) {
      const moduleExports = await import(pathToFileURL(file).href);

      for (const exportName of Object.keys(moduleExports)) {
        if (!Object.hasOwn(transformers, exportName)) {
          missing.push(`${relative(MODELS_DIR, file)}: ${exportName}`);
        }
      }
    }

    expect(missing).toEqual([]);
  });
});

describe("env.backends.onnx.getSupportedDevices", () => {
  const { getSupportedDevices } = transformers.env.backends.onnx;

  it("returns the supported devices for the current environment", () => {
    const devices = getSupportedDevices();
    expect(Array.isArray(devices)).toBe(true);
    expect(devices.length).toBeGreaterThan(0);
    expect(devices).toContain("cpu");
  });

  it("returns a new array on each call, so mutations do not leak", () => {
    const devices = getSupportedDevices();
    expect(devices).not.toBe(getSupportedDevices());

    devices.length = 0;
    expect(getSupportedDevices().length).toBeGreaterThan(0);
  });
});
