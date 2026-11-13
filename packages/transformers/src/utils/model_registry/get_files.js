import { get_tokenizer_files } from './get_tokenizer_files.js';
import { get_model_files } from './get_model_files.js';
import { get_processor_files } from './get_processor_files.js';

/**
 * @typedef {import('../hub.js').PretrainedOptions & Pick<import('../hub.js').ModelSpecificPretrainedOptions, 'dtype'|'device'|'model_file_name'> & {include_tokenizer?: boolean, include_processor?: boolean}} RegistryOptions
 * Loading and file-selection options consumed by the model registry.
 */

/**
 * Returns the list of files that will be loaded for a model based on its configuration.
 * Automatically detects which files are needed (tokenizer, processor, model files).
 *
 * @param {string} modelId The model id (e.g., "Xenova/llama-2-7b")
 * @param {RegistryOptions} [options] Loading and file-selection options.
 * @returns {Promise<string[]>} Array of file paths that will be loaded
 */
export async function get_files(modelId, options = {}) {
    const { include_tokenizer = true, include_processor = true } = options;
    const files = await get_model_files(modelId, options);

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
