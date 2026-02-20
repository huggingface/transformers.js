import { env } from '../../env.js';
import { getCache } from '../../utils/cache.js';
import { isValidUrl } from '../../utils/hub/utils.js';

/**
 * Loads and caches a file from the given URL.
 * @param {string} url The URL of the file to load.
 * @returns {Promise<Response|import('../../utils/hub/files.js').FileResponse|null|string>} The response object, or null if loading failed.
 */
async function loadAndCacheFile(url) {
    const fileName = url.split('/').pop();

    /** @type {import('../../utils/cache.js').CacheInterface|undefined} */
    let cache;
    try {
        cache = await getCache();

        // Try to get from cache first
        if (cache) {
            const result = await cache.match(url);
            if (result) {
                return result;
            }
        }
    } catch (error) {
        console.warn(`Failed to load ${fileName} from cache:`, error);
    }

    // If not in cache, fetch it
    const response = await fetch(url);

    if (!response.ok) {
        throw new Error(`Failed to fetch ${fileName}: ${response.status} ${response.statusText}`);
    }

    // Cache the response for future use
    if (cache) {
        try {
            await cache.put(url, response.clone());
        } catch (e) {
            console.warn(`Failed to cache ${fileName}:`, e);
        }
    }

    return response;
}

/**
 * Loads and caches the WASM binary for ONNX Runtime.
 * @param {string} wasmURL The URL of the WASM file to load.
 * @returns {Promise<ArrayBuffer|null>} The WASM binary as an ArrayBuffer, or null if loading failed.
 */

export async function loadWasmBinary(wasmURL) {
    const response = await loadAndCacheFile(wasmURL);
    if (!response || typeof response === 'string') return null;

    try {
        return await response.arrayBuffer();
    } catch (error) {
        console.warn('Failed to read WASM binary:', error);
        return null;
    }
}

/**
 * Checks if the current environment supports blob URLs for ES modules.
 * Blob URLs don't work in Service Workers, Chrome extensions, with multi-threading, or with COEP headers.
 * @see https://github.com/huggingface/transformers.js/issues/1532
 * @see https://github.com/huggingface/transformers.js/issues/1527
 * @returns {boolean} True if blob URLs are safe to use for module imports.
 */
function canUseBlobURLs() {
    // Don't use blob URLs in Service Workers
    // @ts-ignore - ServiceWorkerGlobalScope may not exist in all environments
    if (typeof ServiceWorkerGlobalScope !== 'undefined' && self instanceof ServiceWorkerGlobalScope) {
        return false;
    }

    // Don't use blob URLs if COEP headers are set (Cross-Origin-Embedder-Policy)
    // COEP blocks cross-origin resources without CORP headers, which breaks CDN loading with blob URLs
    // crossOriginIsolated = true indicates COEP + COOP headers are set
    // @ts-ignore - crossOriginIsolated may not exist in all environments
    if (typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated === true) {
        return false;
    }

    // Don't use blob URLs if workers are being used with threading
    if (env.backends?.onnx?.wasm?.numThreads && env.backends.onnx.wasm.numThreads > 1) {
        return false;
    }

    // Also check global onnxruntime as a fallback
    try {
        // @ts-ignore - onnxruntime may not exist in all environments
        if (typeof globalThis !== 'undefined' && globalThis.onnxruntime?.env?.wasm?.numThreads) {
            // @ts-ignore
            const numThreads = globalThis.onnxruntime.env.wasm.numThreads;
            if (numThreads > 1) {
                return false;
            }
        }
    } catch (e) {
        // Ignore errors checking ONNX config
    }

    // Don't use blob URLs if WASM proxy is enabled (workers spawned to run WASM)
    if (env.backends?.onnx?.wasm?.proxy === true) {
        return false;
    }

    // Check if the current context is a Chrome extension (which may have restrictions)
    // @ts-ignore - chrome may not exist in all environments
    if (typeof chrome !== 'undefined' && chrome.runtime && chrome.runtime.id) {
        return false;
    }

    return true;
}

/**
 * Loads and caches the WASM Factory (.mjs file) for ONNX Runtime.
 * Creates a blob URL from cached content (when safe) to bridge Cache API with dynamic imports used in ORT.
 * Fixes import.meta.url references to point to the correct base URL.
 * @param {string} libURL The URL of the WASM Factory to load.
 * @returns {Promise<string|null>} The blob URL (if enabled), original URL (if disabled), or null if loading failed.
 */
export async function loadWasmFactory(libURL) {
    // Check if we should use blob URLs before doing any work
    const shouldUseBlobURL = env.useWasmBlobURL === true || (env.useWasmBlobURL === 'auto' && canUseBlobURLs());

    // If blob URLs are not safe or disabled, just return the original URL.
    // Don't bother caching since dynamic import() won't use the Cache API anyway.
    if (!shouldUseBlobURL) {
        return libURL;
    }

    // Blob URLs are enabled - fetch from cache or network, then create blob URL
    const response = await loadAndCacheFile(libURL);
    if (!response || typeof response === 'string') return null;

    try {
        let code = await response.text();
        // Fix relative paths when loading factory from blob, overwrite import.meta.url with actual baseURL
        const baseUrl = libURL.split('/').slice(0, -1).join('/');
        code = code.replace(/import\.meta\.url/g, `"${baseUrl}"`);
        const blob = new Blob([code], { type: 'text/javascript' });
        return URL.createObjectURL(blob);
    } catch (error) {
        console.warn('Failed to read WASM factory:', error);
        return null;
    }
}

/**
 * Checks if the given URL is a blob URL (created via URL.createObjectURL).
 * Blob URLs should not be cached as they are temporary in-memory references.
 * @param {string} url - The URL to check.
 * @returns {boolean} True if the URL is a blob URL, false otherwise.
 */
export function isBlobURL(url) {
    return isValidUrl(url, ['blob:']);
}

/**
 * Converts any URL to an absolute URL if needed.
 * If the URL is already absolute (http://, https://, or blob:), returns it unchanged (handled by new URL(...)).
 * Otherwise, resolves it relative to the current page location (browser) or module location (Node/Bun/Deno).
 * @param {string} url - The URL to convert (can be relative or absolute).
 * @returns {string} The absolute URL.
 */
export function toAbsoluteURL(url) {
    let baseURL;

    if (typeof location !== 'undefined' && location.href) {
        // Browser environment: use location.href
        baseURL = location.href;
    } else if (typeof import.meta !== 'undefined' && import.meta.url) {
        // Node.js/Bun/Deno module environment: use import.meta.url
        baseURL = import.meta.url;
    } else {
        // Fallback: if no base is available, return the URL unchanged
        return url;
    }

    return new URL(url, baseURL).href;
}
