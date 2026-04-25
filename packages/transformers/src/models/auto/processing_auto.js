/**
 * @module processors
 */

import { IMAGE_PROCESSOR_NAME } from '../../utils/constants.js';
import { getModelJSON } from '../../utils/hub.js';
import { Processor } from '../../processing_utils.js';

import * as AllProcessors from '../processors.js';
import * as AllImageProcessors from '../image_processors.js';
import * as AllFeatureExtractors from '../feature_extractors.js';

/**
 * @typedef {import('../../processing_utils.js').PretrainedProcessorOptions} PretrainedProcessorOptions
 */

/**
 * Loads a processor from a pretrained id. Unlike `AutoImageProcessor` and
 * `AutoFeatureExtractor`, `AutoProcessor` returns a multi-modal [`Processor`](#processor)
 * that bundles together a tokenizer, image processor, and/or feature extractor
 * — use it when a single model needs more than one.
 *
 * **Example:** Load a Whisper processor (tokenizer + audio feature extractor).
 * ```javascript
 * import { AutoProcessor } from '@huggingface/transformers';
 * const processor = await AutoProcessor.from_pretrained('onnx-community/whisper-tiny.en');
 * ```
 *
 * **Example:** Run an image through a CLIP processor.
 * ```javascript
 * import { AutoProcessor, load_image } from '@huggingface/transformers';
 *
 * const processor = await AutoProcessor.from_pretrained('Xenova/clip-vit-base-patch16');
 * const image = await load_image('https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/football-match.jpg');
 * const { pixel_values } = await processor(image);
 * ```
 */
export class AutoProcessor {
    /** @type {typeof Processor.from_pretrained} */
    static async from_pretrained(pretrained_model_name_or_path, options = {}) {
        // TODO: first check for processor.json
        const preprocessorConfig = await getModelJSON(
            pretrained_model_name_or_path,
            IMAGE_PROCESSOR_NAME,
            true,
            options,
        );

        const { image_processor_type, feature_extractor_type, processor_class } = preprocessorConfig;
        if (processor_class && AllProcessors[processor_class]) {
            return AllProcessors[processor_class].from_pretrained(pretrained_model_name_or_path, options);
        }

        if (!image_processor_type && !feature_extractor_type) {
            throw new Error('No `image_processor_type` or `feature_extractor_type` found in the config.');
        }

        const components = {};
        if (image_processor_type) {
            // Some image processors are saved with the "Fast" suffix, so we remove that if present.
            const image_processor_class = AllImageProcessors[image_processor_type.replace(/Fast$/, '')];
            if (!image_processor_class) {
                throw new Error(`Unknown image_processor_type: '${image_processor_type}'.`);
            }
            components.image_processor = new image_processor_class(preprocessorConfig);
        }

        if (feature_extractor_type) {
            const image_processor_class = AllImageProcessors[feature_extractor_type];
            if (image_processor_class) {
                // Handle legacy case where image processors were specified as feature extractors
                components.image_processor = new image_processor_class(preprocessorConfig);
            } else {
                const feature_extractor_class = AllFeatureExtractors[feature_extractor_type];
                if (!feature_extractor_class) {
                    throw new Error(`Unknown feature_extractor_type: '${feature_extractor_type}'.`);
                }
                components.feature_extractor = new feature_extractor_class(preprocessorConfig);
            }
        }

        const config = {};
        return new Processor(config, components, null);
    }
}
