import { spawnSync } from "node:child_process";

// Exercise the built native entry point with the Tensor implementation installed
// alongside ORT React Native. No mobile bridge or model download is needed.
it("uses the native runtime Tensor when Node and React Native install distinct constructors", () => {
  const script = `
    const assert = require('node:assert/strict');
    const Module = require('node:module');
    const rnRequire = Module.createRequire(require.resolve('onnxruntime-react-native'));
    const nativeCommon = rnRequire('onnxruntime-common');
    // Keep the identities distinct even if future runtime versions are deduped.
    class NativeTensor extends nativeCommon.Tensor {}
    const nativeRuntime = { ...nativeCommon, Tensor: NativeTensor };
    Object.defineProperty(globalThis, 'navigator', { value: { product: 'ReactNative' }, configurable: true });
    const originalLoad = Module._load;
    let transformers;
    try {
      Module._load = function (id, ...args) {
        if (id === 'onnxruntime-react-native') return nativeRuntime;
        if (id === 'native-universal-fs') return { DocumentDirectoryPath: '/tmp', CachesDirectoryPath: '/tmp' };
        if (id === 'react-native') return { Platform: { OS: 'ios' } };
        return originalLoad.call(this, id, ...args);
      };
      transformers = require('./dist/transformers.native.cjs');
    } finally {
      Module._load = originalLoad;
    }
    const tensor = new transformers.Tensor('float32', Float32Array.of(1, 2), [2]);
    assert.ok(tensor.ort_tensor instanceof NativeTensor);
    assert.deepEqual(new transformers.Tensor(tensor.ort_tensor).tolist(), [1, 2]);
    const output = new NativeTensor('float32', Float32Array.of(3, 4), [2]);
    assert.deepEqual(new transformers.Tensor(output).tolist(), [3, 4]);
  `;
  const { status, stderr } = spawnSync(process.execPath, ["-e", script], {
    cwd: new URL("..", import.meta.url),
    encoding: "utf8",
  });
  expect(stderr).toBe("");
  expect(status).toBe(0);
});
