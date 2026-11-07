import { CHAT_TEMPLATE_NAME } from '../constants.js';
import { get_file_metadata } from './get_file_metadata.js';

/**
 * Returns the list of files that will be loaded for a tokenizer.
 * Automatically detects whether the model has tokenizer files.
 *
 * @param {string} modelId The model id to check for tokenizer files
 * @param {import('./get_files.js').RegistryLoadingOptions} [options] Discovery options (no progress callbacks).
 * @returns {Promise<string[]>} An array of file names that will be loaded
 */
export async function get_tokenizer_files(modelId, options = {}) {
    if (!modelId) {
        throw new Error('modelId is required for get_tokenizer_files');
    }

    const [config, chat_template] = await Promise.all([
        get_file_metadata(modelId, 'tokenizer_config.json', options),
        // Newer repos store the chat template in its own file instead of `tokenizer_config.json`.
        get_file_metadata(modelId, CHAT_TEMPLATE_NAME, options).catch((error) => ({ exists: false, error })),
    ]);
    if (!config.exists) {
        return [];
    }
    if ('error' in chat_template && !config.fromCache) {
        // The check failed, which is not the same as the file missing. Only a tokenizer
        // that is served from the cache (e.g. offline) loads without this optional file.
        throw chat_template.error;
    }

    return chat_template.exists
        ? ['tokenizer.json', 'tokenizer_config.json', CHAT_TEMPLATE_NAME]
        : ['tokenizer.json', 'tokenizer_config.json'];
}
