import { AutoFeatureExtractor } from '../auto/feature_extraction_auto.js';
import { AutoTokenizer } from '../auto/tokenization_auto.js';
import { Processor } from '../../processing_utils.js';

/**
 * Processor for Nemo Conformer TDT models.
 */
export class NemoConformerTDTProcessor extends Processor {
    static tokenizer_class = AutoTokenizer;
    static feature_extractor_class = AutoFeatureExtractor;

    /**
     * Preprocess raw audio for Nemo Conformer TDT models.
     * @param {Float32Array|Float64Array} audio
     */
    async _call(audio) {
        return await this.feature_extractor(audio);
    }
}
