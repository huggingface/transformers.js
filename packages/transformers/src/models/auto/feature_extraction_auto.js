/**
 * @module processors
 */

import { FEATURE_EXTRACTOR_NAME, GITHUB_ISSUE_URL } from '../../utils/constants.js';
import { getModelJSON } from '../../utils/hub.js';
import { FeatureExtractor } from '../../feature_extraction_utils.js';
import * as AllFeatureExtractors from '../feature_extractors.js';

/**
 * Loads a feature extractor from a pretrained id. The concrete class is
 * selected from the `feature_extractor_type` in `preprocessor_config.json`.
 * Most commonly used for audio models.
 *
 * ```javascript
 * import { AutoFeatureExtractor, load_audio } from '@huggingface/transformers';
 *
 * const extractor = await AutoFeatureExtractor.from_pretrained('onnx-community/whisper-tiny.en');
 * const audio = await load_audio('https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav', 16000);
 * const { input_features } = await extractor(audio);
 * ```
 */
export class AutoFeatureExtractor {
    /** @type {typeof FeatureExtractor.from_pretrained} */
    static async from_pretrained(pretrained_model_name_or_path, options = {}) {
        const preprocessorConfig = await getModelJSON(
            pretrained_model_name_or_path,
            FEATURE_EXTRACTOR_NAME,
            true,
            options,
        );

        // Determine feature extractor class
        const key = preprocessorConfig.feature_extractor_type;
        const feature_extractor_class = AllFeatureExtractors[key];

        if (!feature_extractor_class) {
            throw new Error(`Unknown feature_extractor_type: '${key}'. Please report this at ${GITHUB_ISSUE_URL}.`);
        }

        // Instantiate feature extractor
        return new feature_extractor_class(preprocessorConfig);
    }
}
