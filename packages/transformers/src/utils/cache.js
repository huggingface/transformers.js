import { apis, env } from '../env.js';
import { FileCache } from './hub/files.js';

/**
 * @typedef {Object} CacheInterface
 * @property {(request: string) => Promise<Response|import('./hub/files.js').FileResponse|undefined|string>} match
 * Checks if a request is in the cache and returns the cached response if found.
 * @property {(request: string, response: Response, progress_callback?: (data: {progress: number, loaded: number, total: number}) => void) => Promise<void>} put
 * Adds a response to the cache.
 */

/**
 * Retrieves an appropriate caching backend based on the environment configuration.
 * Attempts to use custom cache, browser cache, or file system cache in that order of priority.
 * @returns {Promise<CacheInterface | null>}
 * @param file_cache_dir {string|null} Path to a directory in which a downloaded pretrained model configuration should be cached if using the file system cache.
 */
export async function getCache(file_cache_dir = null) {
    // First, check if the a caching backend is available
    // If no caching mechanism available, will download the file every time
    let cache = null;
    if (env.useCustomCache) {
        // Allow the user to specify a custom cache system.
        if (!env.customCache) {
            throw Error('`env.useCustomCache=true`, but `env.customCache` is not defined.');
        }

        // Check that the required methods are defined:
        if (!env.customCache.match || !env.customCache.put) {
            throw new Error(
                '`env.customCache` must be an object which implements the `match` and `put` functions of the Web Cache API. ' +
                    'For more information, see https://developer.mozilla.org/en-US/docs/Web/API/Cache',
            );
        }
        cache = env.customCache;
    }

    if (!cache && env.useBrowserCache) {
        if (typeof caches === 'undefined') {
            throw Error('Browser cache is not available in this environment.');
        }
        try {
            // In some cases, the browser cache may be visible, but not accessible due to security restrictions.
            // For example, when running an application in an iframe, if a user attempts to load the page in
            // incognito mode, the following error is thrown: `DOMException: Failed to execute 'open' on 'CacheStorage':
            // An attempt was made to break through the security policy of the user agent.`
            // So, instead of crashing, we just ignore the error and continue without using the cache.
            cache = await caches.open(env.cacheKey);
        } catch (e) {
            console.warn('An error occurred while opening the browser cache:', e);
        }
    }

    if (!cache && env.useFSCache) {
        if (!apis.IS_FS_AVAILABLE) {
            throw Error('File System Cache is not available in this environment.');
        }

        // If `cache_dir` is not specified, use the default cache directory
        cache = new FileCache(file_cache_dir ?? env.cacheDir);
    }

    return cache;
}

/**
 * Searches the cache for any of the provided names and returns the first match found.
 * @param {CacheInterface} cache The cache to search
 * @param {...string} names The names of the items to search for
 * @returns {Promise<import('./hub/files.js').FileResponse|Response|undefined|string>} The item from the cache, or undefined if not found.
 */
export async function tryCache(cache, ...names) {
    for (let name of names) {
        try {
            let result = await cache.match(name);
            if (result) return result;
        } catch (e) {
            continue;
        }
    }
    return undefined;
}
