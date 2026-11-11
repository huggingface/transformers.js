import { createRequire } from 'node:module';

// Prefer this module's URL: `node -e` can expose the caller's __filename to ESM.
// esbuild replaces import.meta with an empty object in CommonJS, where __filename is local.
const requireFromHere = createRequire(import.meta.url ?? __filename);

/**
 * Loads `onnxruntime-node` on first use, so a runtime injected through `Symbol.for('onnxruntime')`
 * never loads this package's native binding next to the caller's.
 */
export default function loadOnnxNode() {
    // Disable POSIX telemetry before loading the native binding.
    process.env.ORT_DISABLE_TELEMETRY = '1';
    return requireFromHere('onnxruntime-node');
}
