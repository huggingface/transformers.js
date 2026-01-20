import { PreTrainedTokenizer } from '../../tokenization_utils.js';
import { mergeArrays } from '../../utils/core.js';

/**
 * @todo This model is not yet supported by Hugging Face's "fast" tokenizers library (https://github.com/huggingface/tokenizers).
 * Therefore, this implementation (which is based on fast tokenizers) may produce slightly inaccurate results.
 */
export class MarianTokenizer extends PreTrainedTokenizer {
    /**
     * Create a new MarianTokenizer instance.
     * @param {Object} tokenizerJSON The JSON of the tokenizer.
     * @param {Object} tokenizerConfig The config of the tokenizer.
     */
    constructor(tokenizerJSON, tokenizerConfig) {
        super(tokenizerJSON, tokenizerConfig);

        this.languageRegex = /^(>>\w+<<)\s*/g;

        this.supported_language_codes = Array.from(this.get_vocab().keys()).filter((x) => this.languageRegex.test(x));

        console.warn(
            'WARNING: `MarianTokenizer` is not yet supported by Hugging Face\'s "fast" tokenizers library. Therefore, you may experience slightly inaccurate results.',
        );
    }

    /**
     * Encodes a single text. Overriding this method is necessary since the language codes
     * must be removed before encoding with sentencepiece model.
     * @see https://github.com/huggingface/transformers/blob/12d51db243a00726a548a43cc333390ebae731e3/src/transformers/models/marian/tokenization_marian.py#L204-L213
     *
     * @param {string|null} text The text to encode.
     * @returns {string[]|null} The encoded tokens.
     */
    _encode_text(text) {
        if (text === null) return null;

        // Check if text starts with language code:
        const [matchInfo, ...remainder] = text.trim().split(this.languageRegex);

        if (remainder.length === 0) {
            // No language code, encode normally
            return super._encode_text(matchInfo);
        } else if (remainder.length === 2) {
            // Text starts with language code, so we do not encode it with sentencepiece.
            const [language, text] = remainder;

            if (!this.supported_language_codes.includes(language)) {
                console.warn(
                    `Unsupported language code "${language}" detected, which may lead to unexpected behavior. Should be one of: ${JSON.stringify(this.supported_language_codes)}`,
                );
            }
            return mergeArrays([language], super._encode_text(text));
        }
    }
}
