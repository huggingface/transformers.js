import { getModelJSON } from '../hub.js';
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

    // Try to fetch preprocessor_config.json to see if it exists
    const processorConfig = await getModelJSON(modelId, IMAGE_PROCESSOR_NAME, false, {});

    // If file exists, it will have properties; if not, it returns {}
    return Object.keys(processorConfig).length > 0 ? [IMAGE_PROCESSOR_NAME] : [];
}
