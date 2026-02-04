/**
 * @file Model registry for cache and file operations
 *
 * Provides static methods for:
 * - Discovering which files a model needs
 * - Checking cache status
 * - Getting file metadata
 *
 * @module utils/cache
 */

import { get_files } from './get_files.js';
import { get_pipeline_files } from './get_pipeline_files.js';
import { get_model_files } from './get_model_files.js';
import { get_tokenizer_files } from './get_tokenizer_files.js';
import { get_processor_files } from './get_processor_files.js';
import { is_cached, is_pipeline_cached } from './is_cached.js';
import { get_file_metadata } from './get_file_metadata.js';

export class ModelRegistry {
    /**
     * Get all files (model, tokenizer, processor) needed for a model.
     *
     * @param {string} modelId - The model id (e.g., "Xenova/bert-base-uncased")
     * @param {Object} [options] - Optional parameters
     * @param {import('../../configs.js').PretrainedConfig} [options.config=null] - Pre-loaded config
     * @param {import('../dtypes.js').DataType|Record<string, import('../dtypes.js').DataType>} [options.dtype=null] - Override dtype
     * @param {import('../devices.js').DeviceType|Record<string, import('../devices.js').DeviceType>} [options.device=null] - Override device
     * @param {boolean} [options.include_tokenizer=true] - Whether to check for tokenizer files
     * @param {boolean} [options.include_processor=true] - Whether to check for processor files
     * @returns {Promise<string[]>} Array of file paths
     */
    static async get_files(modelId, options = {}) {
        return get_files(modelId, options);
    }

    /**
     * Get all files needed for a specific pipeline task.
     * Automatically determines which components are needed based on the task.
     *
     * @param {string} task - The pipeline task (e.g., "text-generation", "background-removal")
     * @param {string} modelId - The model id (e.g., "Xenova/bert-base-uncased")
     * @param {Object} [options] - Optional parameters
     * @param {import('../../configs.js').PretrainedConfig} [options.config=null] - Pre-loaded config
     * @param {import('../dtypes.js').DataType|Record<string, import('../dtypes.js').DataType>} [options.dtype=null] - Override dtype
     * @param {import('../devices.js').DeviceType|Record<string, import('../devices.js').DeviceType>} [options.device=null] - Override device
     * @returns {Promise<string[]>} Array of file paths
     */
    static async get_pipeline_files(task, modelId, options = {}) {
        return get_pipeline_files(task, modelId, options);
    }

    /**
     * Get model files needed for a specific model.
     *
     * @param {string} modelId - The model id
     * @param {Object} [options] - Optional parameters
     * @param {import('../../configs.js').PretrainedConfig} [options.config=null] - Pre-loaded config
     * @param {import('../dtypes.js').DataType|Record<string, import('../dtypes.js').DataType>} [options.dtype=null] - Override dtype
     * @param {import('../devices.js').DeviceType|Record<string, import('../devices.js').DeviceType>} [options.device=null] - Override device
     * @returns {Promise<string[]>} Array of model file paths
     */
    static async get_model_files(modelId, options = {}) {
        return get_model_files(modelId, options);
    }

    /**
     * Get tokenizer files needed for a specific model.
     *
     * @param {string} modelId - The model id
     * @returns {Promise<string[]>} Array of tokenizer file paths
     */
    static async get_tokenizer_files(modelId) {
        return get_tokenizer_files(modelId);
    }

    /**
     * Get processor files needed for a specific model.
     *
     * @param {string} modelId - The model id
     * @returns {Promise<string[]>} Array of processor file paths
     */
    static async get_processor_files(modelId) {
        return get_processor_files(modelId);
    }

    /**
     * Check if a model and all its required files are cached.
     *
     * @param {string} modelId - The model id
     * @param {Object} [options] - Optional parameters
     * @param {import('../dtypes.js').DataType|Record<string, import('../dtypes.js').DataType>} [options.dtype=null] - Override dtype
     * @param {import('../devices.js').DeviceType|Record<string, import('../devices.js').DeviceType>} [options.device=null] - Override device
     * @returns {Promise<import('./is_cached.js').CacheCheckResult>} Object with allCached boolean and files array with cache status
     *
     * @example
     * // Check cache status
     * const status = await ModelRegistry.is_cached('Xenova/gpt2');
     * console.log(status.allCached ? 'All files cached!' : 'Some files need downloading');
     * console.log(status.files); // [{ file: 'config.json', cached: true }, ...]
     *
     * @example
     * // With options
     * const status = await ModelRegistry.is_cached('Xenova/gpt2', { dtype: 'fp16', device: 'webgpu' });
     * status.files.forEach(f => {
     *     console.log(`${f.file}: ${f.cached ? '✓' : '✗'}`);
     * });
     */
    static async is_cached(modelId, options = {}) {
        return is_cached(modelId, options);
    }

    /**
     * Check if all files for a specific pipeline task are cached.
     * Automatically determines which components are needed based on the task.
     *
     * @param {string} task - The pipeline task (e.g., "text-generation", "background-removal")
     * @param {string} modelId - The model id
     * @param {Object} [options] - Optional parameters
     * @param {string} [options.cache_dir] - Custom cache directory
     * @param {string} [options.revision] - Model revision (default: 'main')
     * @param {import('../../configs.js').PretrainedConfig} [options.config] - Pre-loaded config
     * @param {import('../dtypes.js').DataType|Record<string, import('../dtypes.js').DataType>} [options.dtype=null] - Override dtype
     * @param {import('../devices.js').DeviceType|Record<string, import('../devices.js').DeviceType>} [options.device=null] - Override device
     * @returns {Promise<import('./is_cached.js').CacheCheckResult>} Object with allCached boolean and files array with cache status
     *
     * @example
     * // Check cache status
     * const status = await ModelRegistry.is_pipeline_cached('text-generation', 'Xenova/gpt2');
     * console.log(status.allCached ? 'Ready to use!' : 'Will download files');
     * status.files.forEach(f => {
     *     console.log(`${f.file}: ${f.cached ? '✓' : '✗'}`);
     * });
     *
     * @example
     * // Background removal (only needs model, no tokenizer/processor)
     * const status = await ModelRegistry.is_pipeline_cached('background-removal', 'Xenova/modnet');
     * console.log(`Files needed: ${status.files.length}`); // Should be fewer than full model
     */
    static async is_pipeline_cached(task, modelId, options = {}) {
        return is_pipeline_cached(task, modelId, options);
    }

    /**
     * Get metadata for a specific file without downloading it.
     *
     * @param {string} path_or_repo_id - Model id or path
     * @param {string} filename - The file name
     * @param {import('../hub.js').PretrainedOptions} [options] - Optional parameters
     * @returns {Promise<{exists: boolean, size?: number, contentType?: string, fromCache?: boolean}>} File metadata
     */
    static async get_file_metadata(path_or_repo_id, filename, options = {}) {
        return get_file_metadata(path_or_repo_id, filename, options);
    }
}
