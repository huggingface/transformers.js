/**
 * @file Handler file for choosing the correct version of ONNX Runtime, based on the environment.
 * Ideally, we could import the `onnxruntime-web` and `onnxruntime-node` packages only when needed,
 * but dynamic imports don't seem to work with the current webpack version and/or configuration.
 * This is possibly due to the experimental nature of top-level await statements.
 * So, we just import both packages, and use the appropriate one based on the environment:
 *   - When running in node, we use `onnxruntime-node`.
 *   - When running in the browser, we use `onnxruntime-web` (`onnxruntime-node` is not bundled).
 *
 * This module is not directly exported, but can be accessed through the environment variables:
 * ```javascript
 * import { env } from '@huggingface/transformers';
 * console.log(env.backends.onnx);
 * ```
 *
 * @module backends/onnx
 */

import { env, apis } from '../env.js';

// NOTE: Import order matters here. We need to import `onnxruntime-node` before `onnxruntime-web`.
// In either case, we select the default export if it exists, otherwise we use the named export.
import * as ONNX_NODE from 'onnxruntime-node';
import * as ONNX_WEB from 'onnxruntime-web/webgpu';

export { Tensor } from 'onnxruntime-common';

/**
 * @typedef {import('onnxruntime-common').InferenceSession.ExecutionProviderConfig} ONNXExecutionProviders
 */

/** @type {Record<import("../utils/devices.js").DeviceType, ONNXExecutionProviders>} */
const DEVICE_TO_EXECUTION_PROVIDER_MAPPING = Object.freeze({
    auto: null, // Auto-detect based on device and environment
    gpu: null, // Auto-detect GPU
    cpu: 'cpu', // CPU
    wasm: 'wasm', // WebAssembly
    webgpu: 'webgpu', // WebGPU
    cuda: 'cuda', // CUDA
    dml: 'dml', // DirectML
    coreml: 'coreml', // CoreML

    webnn: { name: 'webnn', deviceType: 'cpu' }, // WebNN (default)
    'webnn-npu': { name: 'webnn', deviceType: 'npu' }, // WebNN NPU
    'webnn-gpu': { name: 'webnn', deviceType: 'gpu' }, // WebNN GPU
    'webnn-cpu': { name: 'webnn', deviceType: 'cpu' }, // WebNN CPU
});

/**
 * The list of supported devices, sorted by priority/performance.
 * @type {import("../utils/devices.js").DeviceType[]}
 */
const supportedDevices = [];

/** @type {ONNXExecutionProviders[]} */
let defaultDevices;
let ONNX;
const ORT_SYMBOL = Symbol.for('onnxruntime');

if (ORT_SYMBOL in globalThis) {
    // If the JS runtime exposes their own ONNX runtime, use it
    ONNX = globalThis[ORT_SYMBOL];
} else if (apis.IS_NODE_ENV) {
    ONNX = ONNX_NODE.default ?? ONNX_NODE;

    // Updated as of ONNX Runtime 1.23.0-dev.20250612-70f14d7670
    // The following table lists the supported versions of ONNX Runtime Node.js binding provided with pre-built binaries.
    // | EPs/Platforms         | Windows x64        | Windows arm64      | Linux x64          | Linux arm64        | MacOS x64          | MacOS arm64        |
    // | --------------------- | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ | ------------------ |
    // | CPU                   | ✔️                  | ✔️                  | ✔️                  | ✔️                  | ✔️                  | ✔️                  |
    // | WebGPU (experimental) | ✔️                  | ✔️                  | ✔️                  | ❌                  | ✔️                  | ✔️                  |
    // | DirectML              | ✔️                  | ✔️                  | ❌                  | ❌                  | ❌                  | ❌                  |
    // | CUDA                  | ❌                  | ❌                  | ✔️ (CUDA v12)       | ❌                  | ❌                  | ❌                  |
    // | CoreML                | ❌                  | ❌                  | ❌                  | ❌                  | ✔️                  | ✔️                  |
    switch (process.platform) {
        case 'win32': // Windows x64 and Windows arm64
            supportedDevices.push('dml');
            break;
        case 'linux': // Linux x64 and Linux arm64
            if (process.arch === 'x64') {
                supportedDevices.push('cuda');
            }
            break;
        case 'darwin': // MacOS x64 and MacOS arm64
            supportedDevices.push('coreml');
            break;
    }

    supportedDevices.push('webgpu');
    supportedDevices.push('cpu');
    defaultDevices = ['cpu'];
} else {
    ONNX = ONNX_WEB;

    if (apis.IS_WEBNN_AVAILABLE) {
        // TODO: Only push supported providers (depending on available hardware)
        supportedDevices.push('webnn-npu', 'webnn-gpu', 'webnn-cpu', 'webnn');
    }

    if (apis.IS_WEBGPU_AVAILABLE) {
        supportedDevices.push('webgpu');
    }

    supportedDevices.push('wasm');
    defaultDevices = ['wasm'];
}

// @ts-ignore
const InferenceSession = ONNX.InferenceSession;

/**
 * Map a device to the execution providers to use for the given device.
 * @param {import("../utils/devices.js").DeviceType|"auto"|null} [device=null] (Optional) The device to run the inference on.
 * @returns {ONNXExecutionProviders[]} The execution providers to use for the given device.
 */
export function deviceToExecutionProviders(device = null) {
    // Use the default execution providers if the user hasn't specified anything
    if (!device) return defaultDevices;

    // Handle overloaded cases
    switch (device) {
        case 'auto':
            return supportedDevices;
        case 'gpu':
            return supportedDevices.filter((x) => ['webgpu', 'cuda', 'dml', 'webnn-gpu'].includes(x));
    }

    if (supportedDevices.includes(device)) {
        return [DEVICE_TO_EXECUTION_PROVIDER_MAPPING[device] ?? device];
    }

    throw new Error(`Unsupported device: "${device}". Should be one of: ${supportedDevices.join(', ')}.`);
}

const IS_WEB_ENV = apis.IS_BROWSER_ENV || apis.IS_WEBWORKER_ENV;

/**
 * Currently, Transformers.js doesn't support simultaneous loading of sessions in WASM/WebGPU.
 * For this reason, we need to chain the loading calls.
 * @type {Promise<any>}
 */
let webInitChain = Promise.resolve();

/**
 * Promise that resolves when WASM binary has been loaded (if caching is enabled).
 * This ensures we only attempt to load the WASM binary once.
 * @type {Promise<void>|null}
 */
let wasmBinaryLoadPromise = null;

/**
 * Ensures the WASM binary is loaded and cached before creating an inference session.
 * Only runs once, even if called multiple times.
 *
 * Note: This caches the WASM binary via the wasmBinary option, which allows it to work offline.
 * However, the MJS loader file still needs to be fetched from the network (or bundled).
 * For full offline support including the MJS file, you need to either:
 * 1. Use a Service Worker to cache all network requests
 * 2. Bundle onnxruntime-web with your application
 *
 * @returns {Promise<void>}
 */
async function ensureWasmBinaryLoaded() {
    // If already loading or loaded, return the existing promise
    if (wasmBinaryLoadPromise) {
        return wasmBinaryLoadPromise;
    }

    // Check if we should load the WASM binary
    if (!env.useWasmCache || !ONNX_ENV?.wasm?.wasmPaths || !IS_WEB_ENV) {
        wasmBinaryLoadPromise = Promise.resolve();
        return wasmBinaryLoadPromise;
    }

    // Start loading the WASM binary
    wasmBinaryLoadPromise = (async () => {
        const urls = getUrlsFromPaths(ONNX_ENV.wasm.wasmPaths);

        // Load and cache the WASM binary
        if (urls.wasm) {
            try {
                const wasmBinary = await loadWasmBinary(urls.wasm);
                if (wasmBinary) {
                    ONNX_ENV.wasm.wasmBinary = wasmBinary;
                }
            } catch (err) {
                console.warn('Failed to pre-load WASM binary:', err);
            }
        }
    })();

    return wasmBinaryLoadPromise;
}

/**
 * Create an ONNX inference session.
 * @param {Uint8Array|string} buffer_or_path The ONNX model buffer or path.
 * @param {import('onnxruntime-common').InferenceSession.SessionOptions} session_options ONNX inference session options.
 * @param {Object} session_config ONNX inference session configuration.
 * @returns {Promise<import('onnxruntime-common').InferenceSession & { config: Object}>} The ONNX inference session.
 */
export async function createInferenceSession(buffer_or_path, session_options, session_config) {
    await ensureWasmBinaryLoaded();

    const load = () => InferenceSession.create(buffer_or_path, session_options);
    const session = await (IS_WEB_ENV ? (webInitChain = webInitChain.then(load)) : load());
    session.config = session_config;
    return session;
}

/**
 * Currently, Transformers.js doesn't support simultaneous execution of sessions in WASM/WebGPU.
 * For this reason, we need to chain the inference calls (otherwise we get "Error: Session already started").
 * @type {Promise<any>}
 */
let webInferenceChain = Promise.resolve();

/**
 * Run an inference session.
 * @param {import('onnxruntime-common').InferenceSession} session The ONNX inference session.
 * @param {Record<string, import('onnxruntime-common').Tensor>} ortFeed The input tensors.
 * @returns {Promise<Record<string, import('onnxruntime-common').Tensor>>} The output tensors.
 */
export async function runInferenceSession(session, ortFeed) {
    const run = () => session.run(ortFeed);
    const output = await (IS_WEB_ENV ? (webInferenceChain = webInferenceChain.then(run)) : run());
    return output;
}

/**
 * Check if an object is an ONNX tensor.
 * @param {any} x The object to check
 * @returns {boolean} Whether the object is an ONNX tensor.
 */
export function isONNXTensor(x) {
    return x instanceof ONNX.Tensor;
}

/**
 * Get the appropriate cache instance based on environment and settings.
 * @returns {Promise<Cache|null>} The cache instance or null if caching is disabled.
 */
async function getWasmCache() {
    if (!env.useWasmCache || !IS_WEB_ENV) {
        return null;
    }

    // Try custom cache first
    if (env.useCustomCache && env.customCache) {
        return env.customCache;
    }

    // Try browser cache (only relevant for web environments)
    if (env.useBrowserCache && typeof caches !== 'undefined') {
        try {
            return await caches.open(env.cacheKey);
        } catch (e) {
            console.warn('Failed to open cache:', e);
        }
    }

    return null;
}

/**
 * Extracts the WASM and MJS file URLs from the wasmPaths configuration.
 * @param {any} wasmPaths The wasmPaths configuration.
 * @returns {{wasm: string|null, mjs: string|null}} Object containing both URLs, or null values if not found.
 */
function getUrlsFromPaths(wasmPaths) {
    if (!wasmPaths) return { wasm: null, mjs: null };

    // If wasmPaths is an object (Safari case), use the wasm and mjs properties
    if (typeof wasmPaths === 'object') {
        return {
            wasm: wasmPaths.wasm ? (wasmPaths.wasm instanceof URL ? wasmPaths.wasm.href : String(wasmPaths.wasm)) : null,
            mjs: wasmPaths.mjs ? (wasmPaths.mjs instanceof URL ? wasmPaths.mjs.href : String(wasmPaths.mjs)) : null,
        };
    }

    // If wasmPaths is a string (prefix), append the appropriate file names
    if (typeof wasmPaths === 'string') {
        // For non-Safari, use asyncify version
        return {
            wasm: `${wasmPaths}ort-wasm-simd-threaded.asyncify.wasm`,
            mjs: `${wasmPaths}ort-wasm-simd-threaded.asyncify.mjs`,
        };
    }

    return { wasm: null, mjs: null };
}

/**
 * Loads and caches a file from the given URL.
 * @param {string} url The URL of the file to load.
 * @returns {Promise<Response|null>} The response object, or null if loading failed.
 */
async function loadAndCacheWasmFile(url) {
    try {
        const cache = await getWasmCache();
        let response;

        // Try to get from cache first
        if (cache) {
            try {
                response = await cache.match(url);
            } catch (e) {
                console.warn(`Error reading wasm file from cache:`, e);
            }
        }

        // If not in cache, fetch it
        if (!response) {
            response = await fetch(url);

            if (!response.ok) {
                throw new Error(`Failed to fetch wasm file: ${response.status} ${response.statusText}`);
            }

            // Cache the response for future use
            if (cache) {
                try {
                    await cache.put(url, response.clone());
                } catch (e) {
                    console.warn(`Failed to cache wasm file:`, e);
                }
            }
        }

        return response;
    } catch (error) {
        console.warn(`Failed to load wasm file:`, error);
        return null;
    }
}

/**
 * Loads and caches the WASM binary for ONNX Runtime.
 * @param {string} wasmURL The URL of the WASM file to load.
 * @returns {Promise<ArrayBuffer|null>} The WASM binary as an ArrayBuffer, or null if loading failed.
 */
async function loadWasmBinary(wasmURL) {
    const response = await loadAndCacheWasmFile(wasmURL);
    if (!response) return null;

    try {
        return await response.arrayBuffer();
    } catch (error) {
        console.warn('Failed to read WASM binary:', error);
        return null;
    }
}

/** @type {import('onnxruntime-common').Env} */
// @ts-ignore
const ONNX_ENV = ONNX?.env;
if (ONNX_ENV?.wasm) {
    // Initialize wasm backend with suitable default settings.

    // (Optional) Set path to wasm files. This will override the default path search behavior of onnxruntime-web.
    // By default, we only do this if we are not in a service worker and the wasmPaths are not already set.
    if (
        // @ts-ignore Cannot find name 'ServiceWorkerGlobalScope'.ts(2304)
        !(typeof ServiceWorkerGlobalScope !== 'undefined' && self instanceof ServiceWorkerGlobalScope) &&
        ONNX_ENV.versions?.web &&
        !ONNX_ENV.wasm.wasmPaths
    ) {
        const wasmPathPrefix = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ONNX_ENV.versions.web}/dist/`;

        ONNX_ENV.wasm.wasmPaths = apis.IS_SAFARI
            ? {
                  mjs: `${wasmPathPrefix}/ort-wasm-simd-threaded.mjs`,
                  wasm: `${wasmPathPrefix}/ort-wasm-simd-threaded.wasm`,
              }
            : wasmPathPrefix;
    }

    // Users may wish to proxy the WASM backend to prevent the UI from freezing,
    // However, this is not necessary when using WebGPU, so we default to false.
    ONNX_ENV.wasm.proxy = false;
}

if (ONNX_ENV?.webgpu) {
    ONNX_ENV.webgpu.powerPreference = 'high-performance';
}

/**
 * Check if ONNX's WASM backend is being proxied.
 * @returns {boolean} Whether ONNX's WASM backend is being proxied.
 */
export function isONNXProxy() {
    // TODO: Update this when allowing non-WASM backends.
    return ONNX_ENV?.wasm?.proxy;
}

// Expose ONNX environment variables to `env.backends.onnx`
env.backends.onnx = ONNX_ENV;
