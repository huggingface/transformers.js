export const DIST_FOLDER = "dist";
export const NODE_IGNORE_MODULES = ["onnxruntime-web"];
export const NODE_EXTERNAL_MODULES = [
  "onnxruntime-common",
  "onnxruntime-node",
  "sharp",
  "node:fs",
  "node:path",
  "node:url",
];

export const WEB_IGNORE_MODULES = ["onnxruntime-node", "sharp", "fs", "path", "url", "stream", "stream/promises"];
export const WEB_EXTERNAL_MODULES = ["onnxruntime-common", "onnxruntime-web"];