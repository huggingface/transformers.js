import { getCache } from '../../utils/cache.js';

/**
 * Loads and caches a file from the given URL.
 * @param {string} url The URL of the file to load.
 * @returns {Promise<Response|import('../../utils/hub/FileResponse.js').default|null>} The response object, or null if loading failed.
 */
async function loadAndCacheFile(url) {
    const fileName = url.split('/').pop();
    try {
        const cache = await getCache();

        // Try to get from cache first
        if (cache) {
            try {
                return await cache.match(url);
            } catch (e) {
                console.warn(`Error reading ${fileName} from cache:`, e);
            }
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
    } catch (error) {
        console.warn(`Failed to load ${fileName}:`, error);
        return null;
    }
}

/**
 * Loads and caches the WASM binary for ONNX Runtime.
 * @param {string} wasmURL The URL of the WASM file to load.
 * @returns {Promise<ArrayBuffer|null>} The WASM binary as an ArrayBuffer, or null if loading failed.
 */

export async function loadWasmBinary(wasmURL) {
    const response = await loadAndCacheFile(wasmURL);
    if (!response) return null;

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
    if (!response) return null;

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
