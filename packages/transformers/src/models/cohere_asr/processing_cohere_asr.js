import { AutoFeatureExtractor } from '../auto/feature_extraction_auto.js';
import { AutoTokenizer } from '../auto/tokenization_auto.js';
import { Processor } from '../../processing_utils.js';

const NO_SPACE_LANGUAGES = new Set(['ja', 'zh']);

export class CohereAsrProcessor extends Processor {
    static tokenizer_class = AutoTokenizer;
    static feature_extractor_class = AutoFeatureExtractor;
    static uses_processor_config = true;

    /**
     * Build the 10-token decoder prompt for the given language.
     * @param {string} [language='en'] Language code.
     * @returns {number[]} Token IDs for the decoder prompt.
     */
    get_decoder_prompt_ids(language = 'en') {
        const tokens = [
            '▁',
            '<|startofcontext|>',
            '<|startoftranscript|>',
            '<|emo:undefined|>',
            `<|${language}|>`,
            `<|${language}|>`,
            '<|pnc|>',
            '<|noitn|>',
            '<|notimestamp|>',
            '<|nodiarize|>',
        ];
        return this.tokenizer.convert_tokens_to_ids(tokens);
    }

    /**
     * Join chunk texts back together, using the appropriate separator for the language.
     * @param {string[]} texts Decoded texts, one per chunk.
     * @param {string} [language='en'] Language code.
     * @returns {string} The joined text.
     */
    static join_chunks(texts, language = 'en') {
        const non_empty = texts.filter((t) => t && t.trim());
        if (non_empty.length === 0) return '';
        const separator = NO_SPACE_LANGUAGES.has(language) ? '' : ' ';
        const parts = [non_empty[0].trimEnd(), ...non_empty.slice(1).map((t) => t.trim())];
        return parts.join(separator);
    }

    /**
     * Calls the feature_extractor function with the given audio input.
     * @param {any} audio The audio input to extract features from.
     * @returns {Promise<any>}
     */
    async _call(audio) {
        return await this.feature_extractor(audio);
    }
}
