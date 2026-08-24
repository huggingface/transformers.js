import * as types from 'node:util/types';
import { onnxruntimeBackend } from 'onnxruntime-node/dist/backend';
import * as ONNX_COMMON from 'onnxruntime-common';

/** Register ONNX Runtime's Node backend in Jest's VM context. */
export function initOnnxTestBackend(): void {
    ONNX_COMMON.env.wasm.numThreads = 1;
    const originalMethod = onnxruntimeBackend.init;
    onnxruntimeBackend.init = function (...args: any[]) {
        Array.isArray = (value: any): value is any[] =>
            typeof value === 'object' &&
            value !== null &&
            typeof value.length === 'number' &&
            value?.constructor.toString() === Array.toString();

        const constructors = [
            'Int8Array',
            'Int16Array',
            'Int32Array',
            'BigInt64Array',
            'Uint8Array',
            'Uint8ClampedArray',
            'Uint16Array',
            'Uint32Array',
            'BigUint64Array',
            'Float16Array',
            'Float32Array',
            'Float64Array',
        ];
        for (const name of constructors) {
            const constructor = (globalThis as any)[name];
            const check = (types as any)[`is${name}`];
            if (!constructor || !check) continue;
            Object.defineProperty(constructor, Symbol.hasInstance, {
                value: check.bind(types),
                writable: true,
                configurable: false,
                enumerable: false,
            });
        }
        return originalMethod.apply(this, args as any);
    };
    ONNX_COMMON.registerBackend('test', onnxruntimeBackend, Number.POSITIVE_INFINITY);
}
