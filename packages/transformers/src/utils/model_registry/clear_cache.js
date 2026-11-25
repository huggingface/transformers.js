/**
 * @file Cache clearing utilities for model files
 *
 * Provides functions to clear cached model files from the cache system.
 */

import { AssetLoadingContext } from '../hub/AssetLoadingContext.js';
import { tryCache } from '../cache.js';
import { get_files } from './get_files.js';
import { get_pipeline_files } from './get_pipeline_files.js';

/**
 * @typedef {Object} FileClearStatus
 * @property {string} file - The file path
 * @property {boolean} deleted - Whether all discovered cache aliases for the file were successfully deleted
 * @property {boolean} wasCached - Whether any current or obsolete cache alias existed before deletion
 */

/**
 * @typedef {Object} CacheClearResult
 * @property {number} filesDeleted - Number of logical files whose discovered cache aliases were all successfully deleted
 * @property {number} filesCached - Number of logical files with at least one current or obsolete cache alias
 * @property {FileClearStatus[]} files - Array of files with their deletion status
 */

/**
 * Internal helper to clear cached files.
 *
 *
 * @private
 * @param {string} modelId - The model id
 * @param {string[]} files - List of file paths to clear
 * @param {Object} options - Options including cache_dir
 * @returns {Promise<CacheClearResult>}
 */
async function clear_files_from_cache(modelId, files, options = {}) {
    const context = new AssetLoadingContext(options);
    const cache = await context.getCache();

    if (!cache) {
        return {
            filesDeleted: 0,
            filesCached: 0,
            files: files.map((filename) => ({ file: filename, deleted: false, wasCached: false })),
        };
    }

    if (!cache.delete) {
        throw new Error('Cache does not support delete operation');
    }

    const results = await Promise.all(
        files.map(async (filename) => {
            const resource = context.resolve(modelId, filename, cache);
            const keys = new Set([...resource.cacheKeys, ...resource.obsoleteCacheKeys]);
            let wasCached = false;
            let allDeleted = true;
            // Loading only accepts current aliases; explicit clearing must also
            // remove stale entries that have never been replaced.
            for (const key of keys) {
                if ((await tryCache(cache, key)) === undefined) continue;
                wasCached = true;
                const deleted = await cache.delete(key);
                allDeleted = deleted && allDeleted;
            }

            return { file: filename, deleted: wasCached && allDeleted, wasCached };
        }),
    );

    return {
        filesDeleted: results.filter((r) => r.deleted).length,
        filesCached: results.filter((r) => r.wasCached).length,
        files: results,
    };
}

/**
 * Clears all cached files for a given model.
 * Automatically determines which files are needed using get_files().
 *
 * @param {string} modelId - The model id (e.g., "Xenova/gpt2")
 * @param {Object} [options] - Optional parameters
 * @param {string} [options.cache_dir] - Custom cache directory
 * @param {string} [options.revision] - Model revision (default: 'main')
 * @param {import('../../configs.js').PretrainedConfig} [options.config] - Pre-loaded config
 * @param {import('../dtypes.js').DataType|Record<string, import('../dtypes.js').DataType>} [options.dtype] - Override dtype
 * @param {import('../devices.js').DeviceType|Record<string, import('../devices.js').DeviceType>} [options.device] - Override device
 * @param {boolean} [options.include_tokenizer=true] - Whether to clear tokenizer files
 * @param {boolean} [options.include_processor=true] - Whether to clear processor files
 * @returns {Promise<CacheClearResult>} Object with deletion statistics and file status
 */
export async function clear_cache(modelId, options = {}) {
    if (!modelId) {
        throw new Error('modelId is required');
    }

    const files = await get_files(modelId, options);
    return await clear_files_from_cache(modelId, files, options);
}

/**
 * Clears all cached files for a specific pipeline task.
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
 * @returns {Promise<CacheClearResult>} Object with deletion statistics and file status
 */
export async function clear_pipeline_cache(task, modelId, options = {}) {
    if (!task) {
        throw new Error('task is required');
    }
    if (!modelId) {
        throw new Error('modelId is required');
    }

    const files = await get_pipeline_files(task, modelId, options);
    return await clear_files_from_cache(modelId, files, options);
}
