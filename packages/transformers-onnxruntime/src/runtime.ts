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

import { getOnnxProviderHost } from './host.js';

declare const __ONNX_MODULE_URL__: string;

// NOTE: Import order matters here. We need to import `onnxruntime-node` before `onnxruntime-web`.
// In either case, we select the default export if it exists, otherwise we use the named export.
import * as ONNX_NODE from 'onnxruntime-node';
import * as ONNX_WEB from 'onnxruntime-web/webgpu';
import type { Env, InferenceSession as OrtInferenceSession, Tensor as OrtTensor } from 'onnxruntime-common';
import { loadWasmBinary, loadWasmFactory } from './wasm-cache.js';
export { Tensor } from 'onnxruntime-common';

const { env, apis, logger } = getOnnxProviderHost();
const LogLevel = { DEBUG: 10, INFO: 20, WARNING: 30, ERROR: 40, NONE: 50 };

function isBlobURL(url: string): boolean {
    return url.startsWith('blob:');
}

function toAbsoluteURL(url: string): string {
    return new URL(url, globalThis.location?.href ?? __ONNX_MODULE_URL__).href;
}

type ExecutionProvider = OrtInferenceSession.ExecutionProviderConfig;

const DEVICE_TO_EXECUTION_PROVIDER_MAPPING: Readonly<Record<string, ExecutionProvider | null>> = Object.freeze({
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
 * Converts any LogLevel value to ONNX Runtime's numeric severity level (0-4).
 * This handles both standard LogLevel values (10, 20, 30, 40, 50) and custom intermediate values.
 *
 * @param {number} logLevel - The LogLevel value to convert
 * @returns {number} ONNX Runtime severity level (0-4)
 */
function getOnnxLogSeverityLevel(logLevel: number): 0 | 1 | 2 | 3 | 4 {
    // ONNX Runtime's log severity levels are defined as follows:
    // (0) ORT_LOGGING_LEVEL_VERBOSE: Print all log messages.
    // (1) ORT_LOGGING_LEVEL_INFO: Print info and higher level log messages.
    // (2) ORT_LOGGING_LEVEL_WARNING: Print warning and higher level log messages.
    // (3) ORT_LOGGING_LEVEL_ERROR: Print error log messages.
    // (4) ORT_LOGGING_LEVEL_FATAL: Print only fatal log messages.
    //
    // In practice, ONNX Runtime's logging is extremely verbose (especially on session creation).
    // For this reason, we map multiple LogLevel values to the same ONNX severity level to avoid
    // overwhelming users with logs.
    if (logLevel <= LogLevel.DEBUG) {
        return 0; // ORT_LOGGING_LEVEL_VERBOSE
    } else if (logLevel <= LogLevel.INFO) {
        return 2; // ORT_LOGGING_LEVEL_WARNING
    } else if (logLevel <= LogLevel.WARNING) {
        return 3; // ORT_LOGGING_LEVEL_ERROR
    } else if (logLevel <= LogLevel.ERROR) {
        return 3; // ORT_LOGGING_LEVEL_ERROR
    } else {
        return 4; // ORT_LOGGING_LEVEL_FATAL
    }
}

const ONNX_LOG_LEVEL_NAMES: Record<0 | 1 | 2 | 3 | 4, 'verbose' | 'info' | 'warning' | 'error' | 'fatal'> = {
    0: 'verbose',
    1: 'info',
    2: 'warning',
    3: 'error',
    4: 'fatal',
};

// The list of supported devices, sorted by priority/performance.
const supportedDevices: string[] = [];

let defaultDevices: ExecutionProvider[];
let ONNX: typeof ONNX_NODE;
const ORT_SYMBOL = Symbol.for('onnxruntime');

if (ORT_SYMBOL in globalThis) {
    // If the JS runtime exposes their own ONNX runtime, use it
    ONNX = (globalThis as any)[ORT_SYMBOL] as typeof ONNX_NODE;
} else if (apis.IS_NODE_ENV) {
    ONNX = ONNX_NODE;

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
    ONNX = ONNX_WEB as unknown as typeof ONNX_NODE;

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

const InferenceSession = ONNX.InferenceSession;

/** Map a device to the execution providers to use for the given device. */
export function deviceToExecutionProviders(device: string | null = null): ExecutionProvider[] {
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

/**
 * Currently, Transformers.js doesn't support simultaneous loading of sessions in WASM/WebGPU.
 * For this reason, we need to chain the loading calls.
 */
let webInitChain: Promise<unknown> = Promise.resolve();

/**
 * Promise that resolves when WASM binary has been loaded (if caching is enabled).
 * This ensures we only attempt to load the WASM binary once.
 */
let wasmLoadPromise: Promise<void> | null = null;

/**
 * Ensures the WASM binary is loaded and cached before creating an inference session.
 * Only runs once, even if called multiple times.
 *
 * @returns {Promise<void>}
 */
async function ensureWasmLoaded() {
    // If already loading or loaded, return the existing promise
    if (wasmLoadPromise) {
        return wasmLoadPromise;
    }

    // Check if we should load the WASM binary
    const shouldUseWasmCache =
        env.useWasmCache &&
        typeof ONNX_ENV?.wasm?.wasmPaths === 'object' &&
        ONNX_ENV?.wasm?.wasmPaths?.wasm &&
        ONNX_ENV?.wasm?.wasmPaths?.mjs;

    if (!shouldUseWasmCache) {
        // In Deno's web runtime, the WASM factory must be loaded via blob URL so that Node.js detection
        // can be patched out (see loadWasmFactory). Without caching, the factory is imported directly
        // from its URL and Deno would crash trying to use Node.js APIs. useWasmCache defaults to true
        // in this environment, so this only happens if the user explicitly disables it.
        if (apis.IS_DENO_WEB_RUNTIME) {
            throw new Error(
                "env.useWasmCache=false is not supported in Deno's web runtime. Remove the useWasmCache override.",
            );
        }
        wasmLoadPromise = Promise.resolve();
        return wasmLoadPromise;
    }

    // Start loading the WASM binary
    wasmLoadPromise = (async () => {
        // At this point, we know wasmPaths is an object (not a string) because
        // shouldUseWasmCache checks for wasmPaths.wasm and wasmPaths.mjs
        const urls = ONNX_ENV.wasm.wasmPaths as { wasm: string; mjs: string };

        // Load both in parallel; the .mjs blob URL is only kept if wasmBinary succeeded.
        // ORT only sets locateFile when wasmBinary is provided (onnxruntime PR https://github.com/microsoft/onnxruntime/pull/27411), which
        // prevents new URL(fileName, import.meta.url) from failing inside a blob URL factory.
        let wasmBinaryLoaded = false;
        await Promise.all([
            // Load and cache the WASM binary
            urls.wasm && !isBlobURL(urls.wasm)
                ? (async () => {
                      try {
                          const wasmBinary = await loadWasmBinary(toAbsoluteURL(urls.wasm));
                          if (wasmBinary) {
                              ONNX_ENV.wasm.wasmBinary = wasmBinary;
                              wasmBinaryLoaded = true;
                          }
                      } catch (err) {
                          logger.warn('Failed to pre-load WASM binary:', err);
                      }
                  })()
                : Promise.resolve(),

            // Load and cache the WASM factory as a blob URL
            urls.mjs && !isBlobURL(urls.mjs)
                ? (async () => {
                      try {
                          const wasmFactoryBlob = await loadWasmFactory(toAbsoluteURL(urls.mjs));
                          if (wasmFactoryBlob) {
                              // @ts-ignore
                              ONNX_ENV.wasm.wasmPaths.mjs = wasmFactoryBlob;
                          }
                      } catch (err) {
                          logger.warn('Failed to pre-load WASM factory:', err);
                      }
                  })()
                : Promise.resolve(),
        ]);

        // If wasmBinary failed to load, revert wasmPaths.mjs to the original URL (factory can only be loaded from blob if ONNX_ENV.wasm.wasmBinary is set. @see ORT PR #27411)
        if (!wasmBinaryLoaded) {
            // @ts-ignore
            ONNX_ENV.wasm.wasmPaths.mjs = urls.mjs;
        }
    })();

    return wasmLoadPromise;
}

/**
 * Create an ONNX inference session.
 * @param {Uint8Array|string} buffer_or_path The ONNX model buffer or path.
 * @param {import('onnxruntime-common').InferenceSession.SessionOptions} session_options ONNX inference session options.
 * @param {Object} session_config ONNX inference session configuration.
 * @returns {Promise<import('onnxruntime-common').InferenceSession & { config: Object }>} The ONNX inference session.
 */
export async function createInferenceSession(
    buffer_or_path: Uint8Array | string,
    session_options: OrtInferenceSession.SessionOptions,
    session_config: Record<string, unknown>,
): Promise<OrtInferenceSession & { config: Record<string, unknown> }> {
    await ensureWasmLoaded();
    const logSeverityLevel = getOnnxLogSeverityLevel(env.logLevel ?? LogLevel.WARNING);
    const load = () => {
        const options = {
            // Set default log severity level, but allow overriding through session options
            logSeverityLevel,
            ...session_options,
        };
        const create = InferenceSession.create as (
            model: Uint8Array | string,
            options: OrtInferenceSession.SessionOptions,
        ) => Promise<OrtInferenceSession>;
        return create(buffer_or_path, options);
    };
    const session = await (apis.IS_WEB_ENV ? (webInitChain = webInitChain.then(load)) : load());
    const configuredSession = session as OrtInferenceSession & { config: Record<string, unknown> };
    configuredSession.config = session_config;
    return configuredSession;
}

/**
 * Currently, Transformers.js doesn't support simultaneous execution of sessions in WASM/WebGPU.
 * For this reason, we need to chain the inference calls (otherwise we get "Error: Session already started").
 */
let webInferenceChain: Promise<Record<string, OrtTensor>> = Promise.resolve({});

/**
 * Run an inference session.
 * @param {import('onnxruntime-common').InferenceSession} session The ONNX inference session.
 * @param {Record<string, import('onnxruntime-common').Tensor>} ortFeed The input tensors.
 * @returns {Promise<Record<string, import('onnxruntime-common').Tensor>>} The output tensors.
 */
export async function runInferenceSession(
    session: OrtInferenceSession,
    ortFeed: Record<string, OrtTensor>,
): Promise<Record<string, OrtTensor>> {
    const run = () => session.run(ortFeed);
    return apis.IS_WEB_ENV ? (webInferenceChain = webInferenceChain.then(run)) : run();
}

/**
 * Check if an object is an ONNX tensor.
 * @param x The object to check
 * @returns {boolean} Whether the object is an ONNX tensor.
 */
export function isONNXTensor(x: unknown): x is OrtTensor {
    return x instanceof ONNX.Tensor;
}
const ONNX_ENV: Env = ONNX.env;

/**
 * Check if ONNX's WASM backend is being proxied.
 * @returns {boolean} Whether ONNX's WASM backend is being proxied.
 */
export function isONNXProxy() {
    // TODO: Update this when allowing non-WASM backends.
    return ONNX_ENV?.wasm?.proxy;
}

if (ONNX_ENV) {
    const configured = (env.backends.onnx ??= {}) as Record<string, any>;
    const configuredWasm = configured.wasm;
    const configuredWebgpu = configured.webgpu;
    for (const [key, value] of Object.entries(configured)) {
        if (key !== 'wasm' && key !== 'webgpu' && key !== 'setLogLevel') {
            (ONNX_ENV as any)[key] = value;
        }
    }

    if (ONNX_ENV.wasm) {
        Object.assign(ONNX_ENV.wasm, configuredWasm ?? {});
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

            let wasmPathSuffix = '.asyncify'; // Default to asyncify WASM build
            if (apis.IS_SAFARI_BELOW_26 && !apis.IS_WEBGPU_AVAILABLE) {
                // Disable asyncify for Safari below 26 when WebGPU is not available
                wasmPathSuffix = '';
            }

            ONNX_ENV.wasm.wasmPaths = {
                mjs: `${wasmPathPrefix}ort-wasm-simd-threaded${wasmPathSuffix}.mjs`,
                wasm: `${wasmPathPrefix}ort-wasm-simd-threaded${wasmPathSuffix}.wasm`,
            };
        }

        // Users may wish to proxy the WASM backend to prevent the UI from freezing,
        // However, this is not necessary when using WebGPU, so we default to false.
        ONNX_ENV.wasm.proxy ??= false;
    }

    if (ONNX_ENV.webgpu) {
        Object.assign(ONNX_ENV.webgpu, configuredWebgpu ?? {});
        ONNX_ENV.webgpu.powerPreference ??= 'high-performance';
    }

    /**
     * A function to map Transformers.js log levels to ONNX Runtime log severity
     * levels, and set the log level environment variable in ONNX Runtime.
     * @param {number} logLevel The log level to set.
     */
    function setLogLevel(logLevel: number) {
        const severityLevel = getOnnxLogSeverityLevel(logLevel);
        ONNX_ENV.logLevel = ONNX_LOG_LEVEL_NAMES[severityLevel];
    }

    // Set the initial log level to be the default Transformers.js log level.
    setLogLevel(env.logLevel ?? LogLevel.WARNING);

    // Expose ONNX environment variables to `env.backends.onnx`
    Object.assign(configured, ONNX_ENV, { setLogLevel });
    configured.wasm = ONNX_ENV.wasm;
    configured.webgpu = ONNX_ENV.webgpu;
    env.backends.onnx = configured;
}
