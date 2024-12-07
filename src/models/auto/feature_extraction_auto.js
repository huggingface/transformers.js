
import { FEATURE_EXTRACTOR_NAME, GITHUB_ISSUE_URL } from '../../utils/constants.js';
import { getModelJSON } from '../../utils/hub.js';
import { FeatureExtractor } from '../../base/feature_extraction_utils.js';
import * as AllFeatureExtractors from '../feature_extractors.js';

export class AutoFeatureExtractor {

    /**
     * Instantiate one of the feature extractor classes of the library from a pretrained model.
     * 
     * The processor class to instantiate is selected based on the `feature_extractor_type` property of
     * the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible)
     * 
     * @param {string} pretrained_model_name_or_path The name or path of the pretrained model. Can be either:
     * - A string, the *model id* of a pretrained processor hosted inside a model repo on huggingface.co.
     *   Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
     *   user or organization name, like `dbmdz/bert-base-german-cased`.
     * - A path to a *directory* containing processor files, e.g., `./my_model_directory/`.
     * @param {import('../../utils/hub.js').PretrainedOptions} options Additional options for loading the processor.
     * 
     * @returns {Promise<AllFeatureExtractors.ImageProcessor>} A new instance of the Processor class.
     */

    /** @type {typeof FeatureExtractor.from_pretrained} */
    static async from_pretrained(pretrained_model_name_or_path, options={}) {

        const preprocessorConfig = await getModelJSON(pretrained_model_name_or_path, FEATURE_EXTRACTOR_NAME, true, options);

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
