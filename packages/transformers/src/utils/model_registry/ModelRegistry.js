/**
 * @file Model registry for cache and file operations
 *
 * Provides static methods for:
 * - Discovering which files a model needs
 * - Getting file metadata
 * - Checking cache status
 *
 * **Example:** Get all files needed for a model
 * ```javascript
 * const files = await ModelRegistry.get_files(
 *   "onnx-community/all-MiniLM-L6-v2-ONNX",
 *   { dtype: "fp16" },
 * );
 * console.log(files); // [ 'config.json', 'onnx/model_fp16.onnx', 'onnx/model_fp16.onnx_data', 'tokenizer.json', 'tokenizer_config.json' ]
 * ```
 *
 * **Example:** Get all files needed for a specific pipeline task
 * ```javascript
 * const files = await ModelRegistry.get_pipeline_files(
 *   "text-generation",
 *   "onnx-community/Qwen3-0.6B-ONNX",
 *   { dtype: "q4" },
 * );
 * console.log(files); // [ 'config.json', 'onnx/model_q4.onnx', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json' ]
 * ```
 *
 * **Example:** Get specific component files
 * ```javascript
 * const modelFiles = await ModelRegistry.get_model_files("onnx-community/all-MiniLM-L6-v2-ONNX", { dtype: "q4" });
 * const tokenizerFiles = await ModelRegistry.get_tokenizer_files("onnx-community/all-MiniLM-L6-v2-ONNX");
 * const processorFiles = await ModelRegistry.get_processor_files("onnx-community/all-MiniLM-L6-v2-ONNX");
 * console.log(modelFiles); // [ 'config.json', 'onnx/model_q4.onnx', 'onnx/model_q4.onnx_data' ]
 * console.log(tokenizerFiles); // [ 'tokenizer.json', 'tokenizer_config.json' ]
 * console.log(processorFiles); // [ ]
 * ```
 *
 * **Example:** Check file metadata without downloading
 * ```javascript
 * const metadata = await ModelRegistry.get_file_metadata(
 *   "onnx-community/Qwen3-0.6B-ONNX",
 *   "config.json"
 * );
 * console.log(metadata); // { exists: true, size: 912, contentType: 'application/json', fromCache: true }
 * ```
 *
 * **Example:** Model cache management
 * ```javascript
 * const modelId = "onnx-community/Qwen3-0.6B-ONNX";
 * const options = { dtype: "q4" };
 *
 * // Check if the model is cached (probably false)
 * let cacheStatus = await ModelRegistry.is_cached(modelId, options);
 * console.log(cacheStatus);
 * // {
 * //   allCached: false,
 * //   files: [ { file: 'config.json', cached: true }, { file: 'onnx/model_q4.onnx', cached: false }, { file: 'generation_config.json', cached: false }, { file: 'tokenizer.json', cached: false }, { file: 'tokenizer_config.json', cached: false } ]
 * // }
 *
 * // Download the model by instantiating a pipeline
 * const generator = await pipeline("text-generation", modelId, options);
 * const output = await generator(
 *   [{ role: "user", content: "What is the capital of France?" }],
 *   { max_new_tokens: 256, do_sample: false },
 * );
 * console.log(output[0].generated_text.at(-1).content); // <think>...</think>\n\nThe capital of France is **Paris**.
 *
 * // Check if the model is cached (should be true now)
 * cacheStatus = await ModelRegistry.is_cached(modelId, options);
 * console.log(cacheStatus);
 * // {
 * //   allCached: true,
 * //   files: [ { file: 'config.json', cached: true }, { file: 'onnx/model_q4.onnx', cached: true }, { file: 'generation_config.json', cached: true }, { file: 'tokenizer.json', cached: true }, { file: 'tokenizer_config.json', cached: true } ]
 * // }
 *
 * // Clear the cache
 * const clearResult = await ModelRegistry.clear_cache(modelId, options);
 * console.log(clearResult);
 * // {
 * //   filesDeleted: 5,
 * //   filesCached: 5,
 * //   files: [ { file: 'config.json', deleted: true, wasCached: true }, { file: 'onnx/model_q4.onnx', deleted: true, wasCached: true }, { file: 'generation_config.json', deleted: true, wasCached: true }, { file: 'tokenizer.json', deleted: true, wasCached: true }, { file: 'tokenizer_config.json', deleted: true, wasCached: true } ]
 * // }
 *
 * // Check if the model is cached (should be false again)
 * cacheStatus = await ModelRegistry.is_cached(modelId, options);
 * console.log(cacheStatus);
 * // {
 * //   allCached: false,
 * //   files: [ { file: 'config.json', cached: true }, { file: 'onnx/model_q4.onnx', cached: false }, { file: 'generation_config.json', cached: false }, { file: 'tokenizer.json', cached: false }, { file: 'tokenizer_config.json', cached: false } ]
 * // }
 * ```
 *
 * @module utils/ModelRegistry
 */

import { get_files } from './get_files.js';
import { get_pipeline_files } from './get_pipeline_files.js';
import { get_model_files } from './get_model_files.js';
import { get_tokenizer_files } from './get_tokenizer_files.js';
import { get_processor_files } from './get_processor_files.js';
import { is_cached, is_pipeline_cached } from './is_cached.js';
import { get_file_metadata } from './get_file_metadata.js';
import { clear_cache, clear_pipeline_cache } from './clear_cache.js';

/**
 * Static class for cache and file management operations.
 * @hideconstructor
 */
export class ModelRegistry {
    /**
     * Get all files (model, tokenizer, processor) needed for a model.
     *
     * @param {string} modelId - The model id (e.g., "onnx-community/bert-base-uncased-ONNX")
     * @param {Object} [options] - Optional parameters
     * @param {import('../../configs.js').PretrainedConfig} [options.config=null] - Pre-loaded config
     * @param {import('../dtypes.js').DataType|Record<string, import('../dtypes.js').DataType>} [options.dtype=null] - Override dtype
     * @param {import('../devices.js').DeviceType|Record<string, import('../devices.js').DeviceType>} [options.device=null] - Override device
     * @param {string} [options.model_file_name=null] - Override the model file name (excluding .onnx suffix)
     * @param {boolean} [options.include_tokenizer=true] - Whether to check for tokenizer files
     * @param {boolean} [options.include_processor=true] - Whether to check for processor files
     * @returns {Promise<string[]>} Array of file paths
     *
     * @example
     * const files = await ModelRegistry.get_files('onnx-community/gpt2-ONNX');
     * console.log(files); // ['config.json', 'tokenizer.json', 'onnx/model_q4.onnx', ...]
     */
    static async get_files(modelId, options = {}) {
        return get_files(modelId, options);
    }

    /**
     * Get all files needed for a specific pipeline task.
     * Automatically determines which components are needed based on the task.
     *
     * @param {string} task - The pipeline task (e.g., "text-generation", "background-removal")
     * @param {string} modelId - The model id (e.g., "onnx-community/bert-base-uncased-ONNX")
     * @param {Object} [options] - Optional parameters
     * @param {import('../../configs.js').PretrainedConfig} [options.config=null] - Pre-loaded config
     * @param {import('../dtypes.js').DataType|Record<string, import('../dtypes.js').DataType>} [options.dtype=null] - Override dtype
     * @param {import('../devices.js').DeviceType|Record<string, import('../devices.js').DeviceType>} [options.device=null] - Override device
     * @param {string} [options.model_file_name=null] - Override the model file name (excluding .onnx suffix)
     * @returns {Promise<string[]>} Array of file paths
     *
     * @example
     * const files = await ModelRegistry.get_pipeline_files('text-generation', 'onnx-community/gpt2-ONNX');
     * console.log(files); // ['config.json', 'tokenizer.json', 'onnx/model_q4.onnx', ...]
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
     * @param {string} [options.model_file_name=null] - Override the model file name (excluding .onnx suffix)
     * @returns {Promise<string[]>} Array of model file paths
     *
     * @example
     * const files = await ModelRegistry.get_model_files('onnx-community/bert-base-uncased-ONNX');
     * console.log(files); // ['config.json', 'onnx/model_q4.onnx', 'generation_config.json']
     */
    static async get_model_files(modelId, options = {}) {
        return get_model_files(modelId, options);
    }

    /**
     * Get tokenizer files needed for a specific model.
     *
     * @param {string} modelId - The model id
     * @returns {Promise<string[]>} Array of tokenizer file paths
     *
     * @example
     * const files = await ModelRegistry.get_tokenizer_files('onnx-community/gpt2-ONNX');
     * console.log(files); // ['tokenizer.json', 'tokenizer_config.json']
     */
    static async get_tokenizer_files(modelId) {
        return get_tokenizer_files(modelId);
    }

    /**
     * Get processor files needed for a specific model.
     *
     * @param {string} modelId - The model id
     * @returns {Promise<string[]>} Array of processor file paths
     *
     * @example
     * const files = await ModelRegistry.get_processor_files('onnx-community/vit-base-patch16-224-ONNX');
     * console.log(files); // ['preprocessor_config.json']
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
     * const status = await ModelRegistry.is_cached('onnx-community/bert-base-uncased-ONNX');
     * console.log(status.allCached); // true or false
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
     * const status = await ModelRegistry.is_pipeline_cached('text-generation', 'onnx-community/gpt2-ONNX');
     * console.log(status.allCached); // true or false
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
     *
     * @example
     * const metadata = await ModelRegistry.get_file_metadata('onnx-community/gpt2-ONNX', 'config.json');
     * console.log(metadata.exists, metadata.size); // true, 665
     */
    static async get_file_metadata(path_or_repo_id, filename, options = {}) {
        return get_file_metadata(path_or_repo_id, filename, options);
    }

    /**
     * Clears all cached files for a given model.
     * Automatically determines which files are needed and removes them from the cache.
     *
     * @param {string} modelId - The model id (e.g., "onnx-community/gpt2-ONNX")
     * @param {Object} [options] - Optional parameters
     * @param {string} [options.cache_dir] - Custom cache directory
     * @param {string} [options.revision] - Model revision (default: 'main')
     * @param {import('../../configs.js').PretrainedConfig} [options.config] - Pre-loaded config
     * @param {import('../dtypes.js').DataType|Record<string, import('../dtypes.js').DataType>} [options.dtype] - Override dtype
     * @param {import('../devices.js').DeviceType|Record<string, import('../devices.js').DeviceType>} [options.device] - Override device
     * @param {boolean} [options.include_tokenizer=true] - Whether to clear tokenizer files
     * @param {boolean} [options.include_processor=true] - Whether to clear processor files
     * @returns {Promise<import('./clear_cache.js').CacheClearResult>} Object with deletion statistics and file status
     *
     * @example
     * const result = await ModelRegistry.clear_cache('onnx-community/bert-base-uncased-ONNX');
     * console.log(`Deleted ${result.filesDeleted} of ${result.filesCached} cached files`);
     */
    static async clear_cache(modelId, options = {}) {
        return clear_cache(modelId, options);
    }

    /**
     * Clears all cached files for a specific pipeline task.
     * Automatically determines which components are needed based on the task.
     *
     * @param {string} task - The pipeline task (e.g., "text-generation", "image-classification")
     * @param {string} modelId - The model id (e.g., "onnx-community/gpt2-ONNX")
     * @param {Object} [options] - Optional parameters
     * @param {string} [options.cache_dir] - Custom cache directory
     * @param {string} [options.revision] - Model revision (default: 'main')
     * @param {import('../../configs.js').PretrainedConfig} [options.config] - Pre-loaded config
     * @param {import('../dtypes.js').DataType|Record<string, import('../dtypes.js').DataType>} [options.dtype] - Override dtype
     * @param {import('../devices.js').DeviceType|Record<string, import('../devices.js').DeviceType>} [options.device] - Override device
     * @returns {Promise<import('./clear_cache.js').CacheClearResult>} Object with deletion statistics and file status
     *
     * @example
     * const result = await ModelRegistry.clear_pipeline_cache('text-generation', 'onnx-community/gpt2-ONNX');
     * console.log(`Deleted ${result.filesDeleted} of ${result.filesCached} cached files`);
     */
    static async clear_pipeline_cache(task, modelId, options = {}) {
        return clear_pipeline_cache(task, modelId, options);
    }
}
