// WORKAROUND: onnxruntime-node's native binding expects Uint16Array for float16 tensors,
// but onnxruntime-common uses Float16Array when available (Node 20+).
// Hide Float16Array before any onnxruntime imports to force Uint16Array usage.
//
// This file runs before any test imports, ensuring Float16Array is hidden
// when onnxruntime-common's checkTypedArray() runs.
//
// TODO: Remove this workaround once onnxruntime-node adds Float16Array support.
// Track the upstream PR: https://github.com/microsoft/onnxruntime/pull/26742
// When that PR is merged and released, this file can be deleted and the
// setupFiles entry in jest.config.mjs can be removed.

if (typeof globalThis.Float16Array !== 'undefined') {
  // Save reference in case other code needs it
  globalThis._Float16Array = globalThis.Float16Array;
  delete globalThis.Float16Array;
}
