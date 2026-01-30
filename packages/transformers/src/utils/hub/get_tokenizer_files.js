import { get_file_metadata } from '../hub.js';

/**
 * Returns the list of files that will be loaded for a tokenizer.
 * Automatically detects whether the model has tokenizer files.
 *
 * @param {string} modelId The model id to check for tokenizer files
 * @returns {Promise<string[]>} An array of file names that will be loaded
 */
export async function get_tokenizer_files(modelId) {
    if (!modelId) {
        throw new Error('modelId is required for get_tokenizer_files');
    }

    const metadata = await get_file_metadata(modelId, 'tokenizer_config.json', {});
    if (metadata.exists) {
        return ['tokenizer.json', 'tokenizer_config.json'];
    }

    return [];
}
