import { env, loadWasmBinary, loadWasmFactory, logger } from '@huggingface/transformers';
import { type LoadBundledLLGuidanceOptions, loadBundledLLGuidance } from 'llguidance';

import { DEFAULT_LLGUIDANCE_WASM_FACTORY_URL, DEFAULT_LLGUIDANCE_WASM_URL } from '../constants';
import { isNodeLikeRuntime } from './runtime';

export type LlguidanceLoadOptions = LoadBundledLLGuidanceOptions & {
    /** Whether to pre-load and cache llguidance WASM assets. Defaults to env.useWasmCache. */
    useWasmCache?: boolean;
};

export async function loadLLGuidance(loadOptions: LlguidanceLoadOptions) {
    const { useWasmCache = env.useWasmCache, ...options } = loadOptions;
    if (!useWasmCache || isNodeLikeRuntime() || options.wasmFactory) {
        return loadBundledLLGuidance(options);
    }

    const wasmSource = options.wasm ?? options.wasmUrl ?? DEFAULT_LLGUIDANCE_WASM_URL;
    const wasmFactorySource = options.wasmFactoryUrl ?? DEFAULT_LLGUIDANCE_WASM_FACTORY_URL;
    const cachedOptions = { ...options };

    const [wasm, wasmFactoryUrl] = await Promise.all([
        loadCacheableWasm(wasmSource),
        loadCacheableWasmFactory(wasmFactorySource),
    ]);

    if (wasm) {
        cachedOptions.wasm = wasm;
        delete cachedOptions.wasmUrl;
    }

    if (wasm && wasmFactoryUrl) {
        cachedOptions.wasmFactoryUrl = wasmFactoryUrl;
    }

    return loadBundledLLGuidance(cachedOptions);
}

async function loadCacheableWasm(source: LoadBundledLLGuidanceOptions['wasm']) {
    const url = toCacheableURL(source);
    if (!url) return null;

    try {
        return await loadWasmBinary(url);
    } catch (error) {
        logger.warn('Failed to pre-load llguidance WASM binary:', error);
        return null;
    }
}

async function loadCacheableWasmFactory(source: LoadBundledLLGuidanceOptions['wasmFactoryUrl']) {
    const url = toCacheableURL(source);
    if (!url) return null;

    try {
        return await loadWasmFactory(url);
    } catch (error) {
        logger.warn('Failed to pre-load llguidance WASM factory:', error);
        return null;
    }
}

function toCacheableURL(source: unknown) {
    if (typeof source === 'string') {
        return isBlobURL(source) ? null : new URL(source, globalThis.location?.href).href;
    }
    if (source instanceof URL) {
        return isBlobURL(source.href) ? null : source.href;
    }
    return null;
}

function isBlobURL(url: string) {
    return url.startsWith('blob:');
}
