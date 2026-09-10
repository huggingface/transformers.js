import { createRequire } from 'node:module';

// Prefer this module's URL: `node -e` can expose the caller's __filename to ESM.
// esbuild replaces import.meta with an empty object in CommonJS, where __filename is local.
const requireFromHere = createRequire(import.meta.url ?? __filename);

// Disable POSIX telemetry before loading the native binding.
process.env.ORT_DISABLE_TELEMETRY = '1';

export default requireFromHere('onnxruntime-node');
