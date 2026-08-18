import { get_tokenizer_files } from './get_tokenizer_files.js';
import { get_model_files } from './get_model_files.js';
import { get_processor_files } from './get_processor_files.js';

/**
 * Returns the list of files that will be loaded for a model based on its configuration.
 * Automatically detects which files are needed (tokenizer, processor, model files).
 *
 * @param {string} modelId The model id (e.g., "Xenova/llama-2-7b")
 * @param {Object} [options] Optional parameters
 * @param {import('../../configs.js').PretrainedConfig} [options.config=null] Pre-loaded model config (optional, will be fetched if not provided)
 * @param {import('../dtypes.js').DataType|Record<string, import('../dtypes.js').DataType>} [options.dtype=null] Override dtype (use this if passing dtype to pipeline)
 * @param {import('../devices.js').DeviceType|Record<string, import('../devices.js').DeviceType>} [options.device=null] Override device (use this if passing device to pipeline)
 * @param {string|null} [options.model_file_name=null|null] Override the model file name (excluding .onnx suffix)
 * @param {boolean} [options.include_tokenizer=true] Whether to check for tokenizer files (set to false for vision-only models)
 * @param {boolean} [options.include_processor=true] Whether to check for processor files
 * @param {string} [options.revision='main'] Git branch, tag, or commit SHA — forwarded, with the other
 * pretrained options below, to every file lookup this makes
 * @param {string} [options.cache_dir] Path to the cache directory
 * @param {boolean} [options.local_files_only=false] Whether to only look for the files locally
 * @returns {Promise<string[]>} Array of file paths that will be loaded
 */
export async function get_files(
    modelId,
    {
        config = null,
        dtype = null,
        device = null,
        model_file_name = null,
        include_tokenizer = true,
        include_processor = true,
        ...options
    } = {},
) {
    // Forward the remaining pretrained options (`revision`, `cache_dir`, ...) to every lookup below: naming a
    // fixed subset here meant a pinned load listed its files from `main`.
    const files = await get_model_files(modelId, { config, dtype, device, model_file_name, ...options });

    if (include_tokenizer) {
        const tokenizerFiles = await get_tokenizer_files(modelId, options);
        files.push(...tokenizerFiles);
    }
    if (include_processor) {
        const processorFiles = await get_processor_files(modelId, options);
        files.push(...processorFiles);
    }

    return files;
}
