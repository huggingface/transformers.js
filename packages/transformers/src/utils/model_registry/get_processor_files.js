import { IMAGE_PROCESSOR_NAME } from '../constants.js';
import { get_file_metadata } from './get_file_metadata.js';

/**
 * Returns the list of processor files that will be loaded for a model.
 * Auto-detects if the model has a processor by checking if preprocessor_config.json exists.
 *
 * @param {string} modelId The model id (e.g., "Xenova/detr-resnet-50")
 * @param {import('../hub.js').PretrainedOptions} [options] Options forwarded to the metadata request, so the
 * existence check reads the same revision the files will be loaded from.
 * @returns {Promise<string[]>} Array of processor file names (empty if no processor)
 */
export async function get_processor_files(modelId, options = {}) {
    if (!modelId) {
        throw new Error('modelId is required');
    }

    // Check if preprocessor_config.json exists
    const metadata = await get_file_metadata(modelId, IMAGE_PROCESSOR_NAME, options);

    return metadata.exists ? [IMAGE_PROCESSOR_NAME] : [];
}
