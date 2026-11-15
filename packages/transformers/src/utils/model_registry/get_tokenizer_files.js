import { get_file_metadata } from './get_file_metadata.js';

/**
 * Returns the list of files that will be loaded for a tokenizer.
 * Automatically detects whether the model has tokenizer files.
 *
 * @param {string} modelId The model id to check for tokenizer files
 * @param {Object} [options] Optional parameters
 * @param {string|null} [options.cache_dir=null] Custom cache directory
 * @param {boolean} [options.local_files_only=false] Only check local files
 * @param {string} [options.revision='main'] Model revision
 * @returns {Promise<string[]>} An array of file names that will be loaded
 */
export async function get_tokenizer_files(
    modelId,
    { cache_dir = null, local_files_only = false, revision = 'main' } = {},
) {
    if (!modelId) {
        throw new Error('modelId is required for get_tokenizer_files');
    }

    const metadata = await get_file_metadata(modelId, 'tokenizer_config.json', {
        cache_dir,
        local_files_only,
        revision,
    });
    if (metadata.exists) {
        return ['tokenizer.json', 'tokenizer_config.json'];
    }

    return [];
}
