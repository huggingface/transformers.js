import { spawnSync } from "child_process";

const MODULE_NAME = "@huggingface/transformers";

/**
 * Import the package in a fresh process, optionally after injecting a runtime through
 * `globalThis[Symbol.for('onnxruntime')]`, and return whether onnxruntime-node's native binding was loaded.
 */
const loadsBinding = (inject) => {
  const code = `
    import { createRequire } from "node:module";
    ${inject ? `globalThis[Symbol.for("onnxruntime")] = { env: {}, Tensor: class {}, InferenceSession: { create() {} } };` : ""}
    await import("${MODULE_NAME}");
    const cache = Object.keys(createRequire(import.meta.url).cache);
    process.stdout.write(String(cache.some((path) => path.endsWith("onnxruntime_binding.node"))));
  `;
  const { status, stdout, stderr } = spawnSync("node", ["--input-type=module", "-e", code]);
  expect(stderr.toString()).toEqual("");
  expect(status).toEqual(0);
  return stdout.toString() === "true";
};

describe("Loading onnxruntime-node", () => {
  it("loads the bundled onnxruntime-node when no runtime is injected", () => {
    expect(loadsBinding(false)).toBe(true);
  });

  it("does not load the bundled onnxruntime-node when a runtime is injected", () => {
    // A second native binding next to the caller's own would break: two onnxruntime-node versions cannot share a
    // process (the second fails to load on a missing VERS_x.y.z symbol).
    expect(loadsBinding(true)).toBe(false);
  });
});
