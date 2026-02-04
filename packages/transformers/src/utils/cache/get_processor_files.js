import { get_file_metadata } from '../hub.js';
import { IMAGE_PROCESSOR_NAME } from '../constants.js';

/**
 * Returns the list of processor files that will be loaded for a model.
 * Auto-detects if the model has a processor by checking if preprocessor_config.json exists.
 *
 * @param {string} modelId The model id (e.g., "Xenova/detr-resnet-50")
 * @returns {Promise<string[]>} Array of processor file names (empty if no processor)
 */
export async function get_processor_files(modelId) {
    if (!modelId) {
        throw new Error('modelId is required');
    }

    // Check if preprocessor_config.json exists
    const metadata = await get_file_metadata(modelId, IMAGE_PROCESSOR_NAME, {});

    return metadata.exists ? [IMAGE_PROCESSOR_NAME] : [];
}
