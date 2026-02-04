import { getCache } from '../cache.js';
import { buildResourcePaths, checkCachedResource } from '../hub.js';
import { get_files } from './get_files.js';

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
 * @returns {Promise<boolean>} True if all files are cached, false otherwise
 *
 * @example
 * import { is_cached } from '@huggingface/transformers';
 *
 * const cached = await is_cached('Xenova/gpt2');
 * console.log(cached ? 'All files cached!' : 'Some files need downloading');
 *
 * // With options
 * const cached2 = await is_cached('Xenova/gpt2', { dtype: 'fp16', device: 'webgpu' });
 */
export async function is_cached(modelId, options = {}) {
    if (!modelId) {
        throw new Error('modelId is required');
    }

    const cache = await getCache(options?.cache_dir);
    if (!cache) {
        return false;
    }

    // Use dynamic import to avoid circular dependency
    const files = await get_files(modelId, options);

    for (const filename of files) {
        const { localPath, proposedCacheKey } = buildResourcePaths(modelId, filename, options, cache);
        const cached = await checkCachedResource(cache, localPath, proposedCacheKey);

        if (!cached) {
            return false;
        }
    }

    return true;
}
