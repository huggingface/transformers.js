/// <reference types="@webgpu/types" />

import { apis } from '../env.js';

import { DEVICE_TYPES } from './devices.js';

// TODO: Use the adapter from `env.backends.onnx.webgpu.adapter` to check for `shader-f16` support,
// when available in https://github.com/microsoft/onnxruntime/pull/19940.
// For more information, see https://github.com/microsoft/onnxruntime/pull/19857#issuecomment-1999984753

/**
 * Checks if WebGPU fp16 support is available in the current environment.
 */
export const isWebGpuFp16Supported = (function () {
    /** @type {boolean} */
    let cachedResult;

    return async function () {
        if (cachedResult === undefined) {
            if (!apis.IS_WEBGPU_AVAILABLE) {
                cachedResult = false;
            } else {
                try {
                    const adapter = await navigator.gpu.requestAdapter();
                    cachedResult = adapter.features.has('shader-f16');
                } catch (e) {
                    cachedResult = false;
                }
            }
        }
        return cachedResult;
    };
})();

export const DATA_TYPES = Object.freeze({
    auto: 'auto', // Auto-detect based on environment
    fp32: 'fp32',
    fp16: 'fp16',
    q8: 'q8',
    int8: 'int8',
    uint8: 'uint8',
    q4: 'q4',
    bnb4: 'bnb4',
    q4f16: 'q4f16', // fp16 model with int4 block weight quantization
});
/** @typedef {keyof typeof DATA_TYPES} DataType */

export const DEFAULT_DEVICE_DTYPE = DATA_TYPES.fp32;
export const DEFAULT_DEVICE_DTYPE_MAPPING = Object.freeze({
    // NOTE: If not specified, will default to fp32
    [DEVICE_TYPES.wasm]: DATA_TYPES.q8,
});

/** @type {Record<Exclude<DataType, "auto">, string>} */
export const DEFAULT_DTYPE_SUFFIX_MAPPING = Object.freeze({
    [DATA_TYPES.fp32]: '',
    [DATA_TYPES.fp16]: '_fp16',
    [DATA_TYPES.int8]: '_int8',
    [DATA_TYPES.uint8]: '_uint8',
    [DATA_TYPES.q8]: '_quantized',
    [DATA_TYPES.q4]: '_q4',
    [DATA_TYPES.q4f16]: '_q4f16',
    [DATA_TYPES.bnb4]: '_bnb4',
});

/**
 * Resolves a dtype configuration value to a concrete dtype string.
 * Handles string, per-file object, and "auto" forms with device-based fallback.
 * @param {DataType|Record<string, DataType>|null|undefined} dtype The dtype config value.
 * @param {string} fileName The model file name to look up if dtype is an object.
 * @param {string} selectedDevice The resolved device string for fallback.
 * @param {DataType|Record<string, DataType>|null} [configDtype=null] Optional config dtype used as fallback when dtype is "auto" (supports device_config overlay in session.js).
 * @returns {DataType} The resolved dtype string.
 */
export function resolveDtype(dtype, fileName, selectedDevice, configDtype = null) {
    /** @type {string|null|undefined} */
    let resolved;
    if (dtype && typeof dtype !== 'string') {
        resolved = dtype.hasOwnProperty(fileName) ? dtype[fileName] : null;
    } else {
        resolved = /** @type {string|null|undefined} */ (dtype);
    }

    // Handle 'auto': try configDtype fallback
    if (resolved === DATA_TYPES.auto) {
        if (configDtype) {
            const fallback = typeof configDtype === 'string' ? configDtype : configDtype?.[fileName];
            if (fallback && fallback !== DATA_TYPES.auto && DATA_TYPES.hasOwnProperty(fallback)) {
                return /** @type {DataType} */ (fallback);
            }
        }
        return DEFAULT_DEVICE_DTYPE_MAPPING[selectedDevice] ?? DEFAULT_DEVICE_DTYPE;
    }

    // Valid known dtype
    if (resolved && DATA_TYPES.hasOwnProperty(resolved)) {
        return /** @type {DataType} */ (resolved);
    }

    // Fallback to device default
    return DEFAULT_DEVICE_DTYPE_MAPPING[selectedDevice] ?? DEFAULT_DEVICE_DTYPE;
}

export const DataTypeMap = Object.freeze({
    float32: Float32Array,
    // @ts-ignore ts(2552) Limited availability of Float16Array across browsers:
    // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Float16Array
    float16: typeof Float16Array !== 'undefined' ? Float16Array : Uint16Array,
    float64: Float64Array,
    string: Array, // string[]
    int8: Int8Array,
    uint8: Uint8Array,
    int16: Int16Array,
    uint16: Uint16Array,
    int32: Int32Array,
    uint32: Uint32Array,
    int64: BigInt64Array,
    uint64: BigUint64Array,
    bool: Uint8Array,
    uint4: Uint8Array,
    int4: Int8Array,
});
