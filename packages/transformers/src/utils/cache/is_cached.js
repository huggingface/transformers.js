import { getCache } from '../cache.js';
import { buildResourcePaths, checkCachedResource } from '../hub.js';
import { get_files } from './get_files.js';
import { get_pipeline_files } from './get_pipeline_files.js';

/**
 * @typedef {Object} FileCacheStatus
 * @property {string} file - The file path
 * @property {boolean} cached - Whether the file is cached
 */

/**
 * @typedef {Object} CacheCheckResult
 * @property {boolean} allCached - Whether all files are cached
 * @property {FileCacheStatus[]} files - Array of files with their cache status
 */

/**
 * Internal helper to check cache status for a list of files
 * @private
 * @param {string} modelId - The model id
 * @param {string[]} files - List of file paths to check
 * @param {Object} options - Options including cache_dir
 * @returns {Promise<CacheCheckResult>}
 */
async function check_files_cache(modelId, files, options = {}) {
    const cache = await getCache(options?.cache_dir);

    const fileStatuses = [];
    let allCached = true;

    if (!cache) {
        // No cache available, all files considered not cached
        for (const filename of files) {
            fileStatuses.push({ file: filename, cached: false });
        }
        return { allCached: false, files: fileStatuses };
    }

    for (const filename of files) {
        const { localPath, proposedCacheKey } = buildResourcePaths(modelId, filename, options, cache);
        const cached = await checkCachedResource(cache, localPath, proposedCacheKey);
        const isCached = !!cached;

        fileStatuses.push({ file: filename, cached: isCached });

        if (!isCached) {
            allCached = false;
        }
    }

    return { allCached, files: fileStatuses };
}

/**
 * Checks if all files for a given model are already cached.
 * Automatically determines which files are needed using get_files().
 *
 * @param {string} modelId The model id (e.g., "Xenova/gpt2")
 * @param {Object} [options] Optional parameters
 * @param {string} [options.cache_dir] Custom cache directory
 * @param {string} [options.revision] Model revision (default: 'main')
 * @param {import('../../configs.js').PretrainedConfig} [options.config] Pre-loaded config
 * @param {import('../dtypes.js').DataType|Record<string, import('../dtypes.js').DataType>} [options.dtype] Override dtype
 * @param {import('../devices.js').DeviceType|Record<string, import('../devices.js').DeviceType>} [options.device] Override device
 * @returns {Promise<CacheCheckResult>} Object with allCached boolean and files array with cache status
 */
export async function is_cached(modelId, options = {}) {
    if (!modelId) {
        throw new Error('modelId is required');
    }

    const files = await get_files(modelId, options);
    return await check_files_cache(modelId, files, options);
}

/**
 * Checks if all files for a specific pipeline task are already cached.
 * Automatically determines which components are needed based on the task.
 *
 * @param {string} task - The pipeline task (e.g., "text-generation", "image-classification")
 * @param {string} modelId - The model id (e.g., "Xenova/gpt2")
 * @param {Object} [options] - Optional parameters
 * @param {string} [options.cache_dir] - Custom cache directory
 * @param {string} [options.revision] - Model revision (default: 'main')
 * @param {import('../../configs.js').PretrainedConfig} [options.config] - Pre-loaded config
 * @param {import('../dtypes.js').DataType|Record<string, import('../dtypes.js').DataType>} [options.dtype] - Override dtype
 * @param {import('../devices.js').DeviceType|Record<string, import('../devices.js').DeviceType>} [options.device] - Override device
 * @returns {Promise<CacheCheckResult>} Object with allCached boolean and files array with cache status
 */
export async function is_pipeline_cached(task, modelId, options = {}) {
    if (!task) {
        throw new Error('task is required');
    }
    if (!modelId) {
        throw new Error('modelId is required');
    }

    const files = await get_pipeline_files(task, modelId, options);
    return await check_files_cache(modelId, files, options);
}
