import { get_file_metadata } from './get_file_metadata.js';

/**
 * Returns the list of files that will be loaded for a tokenizer.
 * Automatically detects whether the model has tokenizer files.
 *
 * @param {string} modelId The model id to check for tokenizer files
 * @param {import('../hub.js').PretrainedOptions} [options] An object containing optional parameters.
 * @returns {Promise<string[]>} An array of file names that will be loaded
 */
export async function get_tokenizer_files(modelId, options = {}) {
    if (!modelId) {
        throw new Error('modelId is required for get_tokenizer_files');
    }

    const metadata = await get_file_metadata(modelId, 'tokenizer_config.json', options);
    if (metadata.exists) {
        return ['tokenizer.json', 'tokenizer_config.json'];
    }

    return [];
}
