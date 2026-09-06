// Run the browser WebGPU runtime through Dawn's Node binding for this benchmark.
// Source imports of onnxruntime-node must resolve to that same runtime instance.
let runtime;
export function initialize(data) {
  runtime = data?.runtime;
}
export function resolve(specifier, context, nextResolve) {
  if (specifier === "onnxruntime-node" || (runtime && specifier === "onnxruntime-web/webgpu")) {
    return nextResolve(runtime ?? "onnxruntime-web/webgpu", context);
  }
  return nextResolve(specifier, context);
}
