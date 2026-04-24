/**
 * @module processors
 */

import { getModelJSON } from '../../utils/hub.js';
import { ImageProcessor } from '../../image_processors_utils.js';
import * as AllImageProcessors from '../image_processors.js';
import { GITHUB_ISSUE_URL, IMAGE_PROCESSOR_NAME } from '../../utils/constants.js';
import { logger } from '../../utils/logger.js';

/**
 * Helper class which is used to instantiate pretrained image processors with the `from_pretrained` function.
 * The chosen image processor class is determined by the type specified in the preprocessor config.
 *
 * **Example:**
 * ```javascript
 * import { AutoImageProcessor, load_image } from '@huggingface/transformers';
 *
 * const processor = await AutoImageProcessor.from_pretrained('Xenova/clip-vit-base-patch16');
 * const image = await load_image('https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/artemis.jpeg');
 * const { pixel_values } = await processor(image);
 * ```
 */
export class AutoImageProcessor {
    /** @type {typeof ImageProcessor.from_pretrained} */
    static async from_pretrained(pretrained_model_name_or_path, options = {}) {
        const preprocessorConfig = await getModelJSON(
            pretrained_model_name_or_path,
            IMAGE_PROCESSOR_NAME,
            true,
            options,
        );

        // Determine image processor class
        const key = preprocessorConfig.image_processor_type ?? preprocessorConfig.feature_extractor_type;
        let image_processor_class = AllImageProcessors[key?.replace(/Fast$/, '')];

        if (!image_processor_class) {
            if (key !== undefined) {
                // Only log a warning if the class is not found and the key is set.
                logger.warn(
                    `Image processor type '${key}' not found, assuming base ImageProcessor. Please report this at ${GITHUB_ISSUE_URL}.`,
                );
            }
            image_processor_class = ImageProcessor;
        }

        // Instantiate image processor
        return new image_processor_class(preprocessorConfig);
    }
}
