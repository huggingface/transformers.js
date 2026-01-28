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
 * Loads and caches the WASM Factory for ONNX Runtime.
 * @param {string} libURL The URL of the WASM Factory to load.
 * @returns {Promise<string|null>} The blob URL of the WASM Factory, or null if loading failed.
 */
export async function loadWasmFactory(libURL) {
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
        console.warn('Failed to read WASM binary:', error);
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
