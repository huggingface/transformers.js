import { apis } from '../env.js';

/**
 * The list of devices supported by Transformers.js
 */
export const DEVICE_TYPES = Object.freeze({
    auto: 'auto', // Auto-detect based on device and environment
    gpu: 'gpu', // Auto-detect GPU
    cpu: 'cpu', // CPU
    wasm: 'wasm', // WebAssembly
    webgpu: 'webgpu', // WebGPU
    cuda: 'cuda', // CUDA
    dml: 'dml', // DirectML
    coreml: 'coreml', // CoreML

    webnn: 'webnn', // WebNN (default)
    'webnn-npu': 'webnn-npu', // WebNN NPU
    'webnn-gpu': 'webnn-gpu', // WebNN GPU
    'webnn-cpu': 'webnn-cpu', // WebNN CPU
});
const DEFAULT_DEVICE = apis.IS_NODE_ENV ? 'cpu' : 'wasm';

/**
 * @typedef {keyof typeof DEVICE_TYPES} DeviceType
 */

/**
 * Resolves a device configuration value to a concrete device string.
 * Handles string, per-file object, or undefined config, with a default fallback.
 * @param {string|Record<string, string>|null|undefined} deviceConfig The device config value.
 * @param {string} fileName The model file name to look up if deviceConfig is an object.
 * @param {Object} [options]
 * @param {(message: string) => void} [options.warn] Optional callback invoked when deviceConfig is a per-file object but fileName is not found.
 * @returns {string} The resolved device string.
 */
export function selectDevice(deviceConfig, fileName, { warn } = {}) {
    if (!deviceConfig) return DEFAULT_DEVICE;
    if (typeof deviceConfig === 'string') return deviceConfig;
    if (deviceConfig.hasOwnProperty(fileName)) return deviceConfig[fileName];
    if (warn) {
        warn(`device not specified for "${fileName}". Using the default device (${DEFAULT_DEVICE}).`);
    }
    return DEFAULT_DEVICE;
}
