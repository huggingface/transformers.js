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
 * @param {string|null} [options.cache_dir=null] Custom cache directory
 * @param {boolean} [options.local_files_only=false] Never hit the network if true
 * @param {string} [options.revision='main'] Model revision
 * @param {import('../../backends/model_registry.js').ModelRegistryInferenceProvider|null} [options.inferenceProvider=null] Artifact metadata provider
 * @param {boolean} [options.include_tokenizer=true] Whether to check for tokenizer files (set to false for vision-only models)
 * @param {boolean} [options.include_processor=true] Whether to check for processor files
 * @param {boolean} [options.include_model=true] Whether to include built-in ONNX model files
 * @returns {Promise<string[]>} Array of file paths that will be loaded
 */
export async function get_files(
    modelId,
    {
        config = null,
        dtype = null,
        device = null,
        model_file_name = null,
        cache_dir = null,
        local_files_only = false,
        revision = 'main',
        inferenceProvider = null,
        include_tokenizer = true,
        include_processor = true,
        include_model = true,
    } = {},
) {
    const metadataOptions = { cache_dir, local_files_only, revision };
    const files = include_model
        ? await get_model_files(modelId, {
              config,
              dtype,
              device,
              model_file_name,
              inferenceProvider,
              ...metadataOptions,
          })
        : [];

    if (include_tokenizer) {
        const tokenizerFiles = await get_tokenizer_files(modelId, metadataOptions);
        files.push(...tokenizerFiles);
    }
    if (include_processor) {
        const processorFiles = await get_processor_files(modelId, metadataOptions);
        files.push(...processorFiles);
    }

    return files;
}
