import { createRequire } from 'node:module';

// Resolve from this file in both ESM and CommonJS builds.
const requireFromHere = createRequire(typeof __filename === 'string' ? __filename : import.meta.url);

// Disable POSIX telemetry before loading the native binding.
process.env.ORT_DISABLE_TELEMETRY = '1';

export default requireFromHere('onnxruntime-node');
