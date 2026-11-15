import { IMAGE_PROCESSOR_NAME } from '../constants.js';
import { get_file_metadata } from './get_file_metadata.js';

/**
 * Returns the list of processor files that will be loaded for a model.
 * Auto-detects if the model has a processor by checking if preprocessor_config.json exists.
 *
 * @param {string} modelId The model id (e.g., "Xenova/detr-resnet-50")
 * @param {Object} [options] Optional parameters
 * @param {string|null} [options.cache_dir=null] Custom cache directory
 * @param {boolean} [options.local_files_only=false] Only check local files
 * @param {string} [options.revision='main'] Model revision
 * @returns {Promise<string[]>} Array of processor file names (empty if no processor)
 */
export async function get_processor_files(
    modelId,
    { cache_dir = null, local_files_only = false, revision = 'main' } = {},
) {
    if (!modelId) {
        throw new Error('modelId is required');
    }

    // Check if preprocessor_config.json exists
    const metadata = await get_file_metadata(modelId, IMAGE_PROCESSOR_NAME, {
        cache_dir,
        local_files_only,
        revision,
    });

    return metadata.exists ? [IMAGE_PROCESSOR_NAME] : [];
}
