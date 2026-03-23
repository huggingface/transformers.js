import { apis, env } from '../../env.js';
import { getCache } from '../../utils/cache.js';
import { logger } from '../../utils/logger.js';

/**
 * Loads and caches a file from the given URL.
 * @param {string} url The URL of the file to load.
 * @returns {Promise<Response|import('../../utils/hub/FileResponse.js').FileResponse|null|string>} The response object, or null if loading failed.
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
        logger.warn(`Failed to load ${fileName} from cache:`, error);
    }

    // If not in cache, fetch it
    const response = await env.fetch(url);

    if (!response.ok) {
        throw new Error(`Failed to fetch ${fileName}: ${response.status} ${response.statusText}`);
    }

    // Cache the response for future use
    if (cache) {
        try {
            await cache.put(url, response.clone());
        } catch (e) {
            logger.warn(`Failed to cache ${fileName}:`, e);
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
        logger.warn('Failed to read WASM binary:', error);
        return null;
    }
}

/**
 * Loads and caches the WASM Factory (.mjs file) for ONNX Runtime.
 * Creates a blob URL from cached content (when safe) to bridge Cache API with dynamic imports used in ORT.
 * @param {string} libURL The URL of the WASM Factory to load.
 * @returns {Promise<string|null>} The blob URL (if enabled), original URL (if disabled), or null if loading failed.
 */
export async function loadWasmFactory(libURL) {
    // We can't use Blob URLs in some environments (Service Workers, Chrome extensions) due to security restrictions on dynamic import() of blob URLs.
    // In such cases, just return the original URL and don't bother caching since dynamic import() won't use the Cache API anyway.
    // See https://github.com/huggingface/transformers.js/issues/1532.
    if (apis.IS_SERVICE_WORKER_ENV || apis.IS_CHROME_AVAILABLE) {
        return libURL;
    }

    // Fetch from cache or network, then create blob URL
    const response = await loadAndCacheFile(libURL);
    if (!response || typeof response === 'string') return null;

    try {
        let code = await response.text();

        // Handle the case where we are importing the bundled version of the library in Deno (e.g., via CDN or local file),
        // where we need to patch out Node.js detection in the factory. Without this, Deno (which exposes globalThis.process.versions.node)
        // would enter the Node.js branch and try to use Node.js APIs (worker_threads, fs, etc.) that aren't used in the bundled web version.
        // Only needed for the asyncify (single-threaded) variant loaded via blob URL. The module-level pthread auto-start code is unreachable since asyncify never spawns workers.
        // See https://github.com/huggingface/transformers.js/pull/1546/ for more information.
        //
        // NOTE: This does not affect default usage via Deno (i.e., imported via npm: prefix), since we'll be using onnxruntime-node (Native) instead of onnxruntime-web (WASM).
        code = code.replaceAll('globalThis.process?.versions?.node', 'false');
        const blob = new Blob([code], { type: 'text/javascript' });
        return URL.createObjectURL(blob);
    } catch (error) {
        logger.warn('Failed to read WASM factory:', error);
        return null;
    }
}
