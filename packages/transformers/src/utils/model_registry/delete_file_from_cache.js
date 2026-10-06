import { getCache } from '../cache.js';
import { buildResourcePaths, checkCachedResource } from '../hub.js';

/**
 * Delete one file from Transformers.js cache storage.
 *
 * @param {string} modelId
 * @param {string} filename
 * @param {Object} [options]
 * @returns {Promise<boolean>}
 */
export async function delete_file_from_cache(modelId, filename, options = {}) {
    const cache = await getCache(options?.cache_dir);
    if (!cache) return false;
    if (!cache.delete) throw new Error('Cache does not support delete operation');

    const { localPath, proposedCacheKey } = buildResourcePaths(modelId, filename, options, cache);
    const cached = await checkCachedResource(cache, localPath, proposedCacheKey);
    if (!cached) return false;

    const deletedWithProposed = await cache.delete(proposedCacheKey);
    return deletedWithProposed || (proposedCacheKey !== localPath && (await cache.delete(localPath)));
}
