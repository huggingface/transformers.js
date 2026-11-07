import { CHAT_TEMPLATE_NAME } from '../constants.js';
import { get_file_metadata } from './get_file_metadata.js';

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

    const [config, chat_template] = await Promise.all([
        get_file_metadata(modelId, 'tokenizer_config.json', {}),
        get_file_metadata(modelId, CHAT_TEMPLATE_NAME, {}),
    ]);
    if (!config.exists) {
        return [];
    }

    // Newer repos store the chat template in its own file instead of `tokenizer_config.json`.
    return chat_template.exists
        ? ['tokenizer.json', 'tokenizer_config.json', CHAT_TEMPLATE_NAME]
        : ['tokenizer.json', 'tokenizer_config.json'];
}
