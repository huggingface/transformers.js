/**
 * @file Cache clearing utilities for model files
 *
 * Provides functions to clear cached model files from the cache system.
 *
 * @module utils/cache
 */

import { getCache } from '../cache.js';
import { buildResourcePaths, checkCachedResource } from '../hub.js';
import { get_files } from './get_files.js';
import { get_pipeline_files } from './get_pipeline_files.js';

/**
 * @typedef {Object} FileClearStatus
 * @property {string} file - The file path
 * @property {boolean} deleted - Whether the file was successfully deleted
 * @property {boolean} wasCached - Whether the file was cached before deletion
 */

/**
 * @typedef {Object} CacheClearResult
 * @property {number} filesDeleted - Number of files successfully deleted
 * @property {number} filesCached - Number of files that were in cache
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
    const cache = await getCache(options?.cache_dir);

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
            const { localPath, proposedCacheKey } = buildResourcePaths(modelId, filename, options, cache);

            // Check if file is cached
            const cached = await checkCachedResource(cache, localPath, proposedCacheKey);
            const wasCached = !!cached;

            // Try to delete using both possible cache keys
            let deleted = false;
            if (wasCached) {
                // Try deleting with proposedCacheKey first (most likely to work)
                const deletedWithProposed = await cache.delete(proposedCacheKey);

                // If that didn't work and keys are different, try localPath
                const deletedWithLocal = proposedCacheKey !== localPath ? await cache.delete(localPath) : false;

                deleted = deletedWithProposed || deletedWithLocal;
            }

            return { file: filename, deleted, wasCached };
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
 *
 * @example
 * import { clear_cache } from '@huggingface/transformers';
 *
 * // Clear all cached files for a model
 * const result = await clear_cache('Xenova/gpt2');
 * console.log(`Deleted ${result.filesDeleted} of ${result.filesCached} cached files`);
 * result.files.forEach(f => {
 *     console.log(`${f.file}: ${f.deleted ? '✓ deleted' : f.wasCached ? '✗ failed' : 'not cached'}`);
 * });
 *
 * @example
 * // Clear only specific dtype/device variant
 * const result = await clear_cache('Xenova/gpt2', { dtype: 'fp16', device: 'webgpu' });
 * console.log(`Cleared ${result.filesDeleted} files`);
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
 *
 * @example
 * import { clear_pipeline_cache } from '@huggingface/transformers';
 *
 * // Clear cached files for a specific pipeline
 * const result = await clear_pipeline_cache('text-generation', 'Xenova/gpt2');
 * console.log(`Deleted ${result.filesDeleted} of ${result.filesCached} cached files`);
 * result.files.forEach(f => {
 *     console.log(`${f.file}: ${f.deleted ? '✓ deleted' : f.wasCached ? '✗ failed' : 'not cached'}`);
 * });
 *
 * @example
 * // Background removal (only clears model files, no tokenizer/processor)
 * const result = await clear_pipeline_cache('background-removal', 'Xenova/modnet');
 * console.log(`Cleared ${result.filesDeleted} files`);
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
