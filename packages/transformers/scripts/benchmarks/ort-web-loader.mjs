// Run the browser WebGPU runtime through Dawn's Node binding for this benchmark.
// Source imports of onnxruntime-node must resolve to that same runtime instance.
export function resolve(specifier, context, nextResolve) {
  return nextResolve(specifier === "onnxruntime-node" ? "onnxruntime-web/webgpu" : specifier, context);
}
