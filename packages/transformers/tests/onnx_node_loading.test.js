import { spawnSync } from "child_process";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const MODULE_NAME = "@huggingface/transformers";
const WEB_BUNDLE = pathToFileURL(join(dirname(fileURLToPath(import.meta.url)), "..", "dist", "transformers.web.js")).href;

const INJECT = `globalThis[Symbol.for("onnxruntime")] = { env: {}, Tensor: class {}, InferenceSession: { create() {} } };`;
const REPORT = `process.stdout.write(String(Object.keys(require.cache).some((path) => path.endsWith("onnxruntime_binding.node"))));`;

const LOADERS = {
  esm: {
    args: ["--input-type=module"],
    load: `import { createRequire } from "node:module"; const require = createRequire(import.meta.url); await import("${MODULE_NAME}");`,
  },
  cjs: {
    args: [],
    load: `require("${MODULE_NAME}");`,
  },
  web: {
    args: ["--input-type=module"],
    load: `import { createRequire } from "node:module"; const require = createRequire(import.meta.url); await import("${WEB_BUNDLE}");`,
  },
};

/**
 * Load the package in a fresh process, optionally after injecting a runtime through
 * `globalThis[Symbol.for('onnxruntime')]`, and return whether onnxruntime-node's native binding was loaded.
 */
const loadsBinding = (loader, inject) => {
  const { args, load } = LOADERS[loader];
  const code = `${inject ? INJECT : ""}${load}${REPORT}`;
  const { status, stdout, stderr } = spawnSync("node", [...args, "-e", code]);
  expect(stderr.toString()).toEqual("");
  expect(status).toEqual(0);
  return stdout.toString() === "true";
};

describe("Loading onnxruntime-node", () => {
  describe.each(["esm", "cjs"])("node bundle (%s)", (loader) => {
    it("loads the bundled onnxruntime-node when no runtime is injected", () => {
      expect(loadsBinding(loader, false)).toBe(true);
    });

    it("does not load the bundled onnxruntime-node when a runtime is injected", () => {
      // A second native binding next to the caller's own would break: two onnxruntime-node versions cannot share a
      // process (the second fails to load on a missing VERS_x.y.z symbol).
      expect(loadsBinding(loader, true)).toBe(false);
    });
  });

  describe("web bundle", () => {
    it.each([false, true])("imports without onnxruntime-node (runtime injected: %s)", (inject) => {
      expect(loadsBinding("web", inject)).toBe(false);
    });
  });
});
