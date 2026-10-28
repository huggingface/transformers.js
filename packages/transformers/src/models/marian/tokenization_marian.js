import { PreTrainedTokenizer } from '../../tokenization_utils.js';
import { logger } from '../../utils/logger.js';

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

        this.languageRegex = /^(>>\w+<<)\s*/;

        this.supported_language_codes = Array.from(this.get_vocab().keys()).filter((x) => this.languageRegex.test(x));

        logger.warn(
            'WARNING: `MarianTokenizer` is not yet supported by Hugging Face\'s "fast" tokenizers library. Therefore, you may experience slightly inaccurate results.',
        );
    }

    /**
     * Splits the language code (e.g., `>>fr<<`) off the start of the text, since it
     * must be removed before encoding with the sentencepiece model.
     * @see https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/marian/tokenization_marian.py#L185-L196
     *
     * @param {string} text The text to split.
     * @returns {[string|null, string]} The language code (or `null` if the text does not start with one) and the remaining text.
     * @private
     */
    _split_language_code(text) {
        const match = text.match(this.languageRegex);
        if (match === null) {
            return [null, text];
        }
        const language = match[1];
        if (!this.supported_language_codes.includes(language)) {
            logger.warn(
                `Unsupported language code "${language}" detected, which may lead to unexpected behavior. Should be one of: ${JSON.stringify(this.supported_language_codes)}`,
            );
        }
        return [language, text.slice(match[0].length)];
    }

    /**
     * @type {PreTrainedTokenizer['_encode_plus']}
     * @protected
     */
    _encode_plus(text, options) {
        const [language, remainder] = this._split_language_code(text);
        const encoded = super._encode_plus(remainder, options);
        if (language !== null) {
            encoded.input_ids.unshift(this.convert_tokens_to_ids(language) ?? this.unk_token_id);
            encoded.attention_mask.unshift(1);
            encoded.token_type_ids?.unshift(0);
        }
        return encoded;
    }

    /** @type {PreTrainedTokenizer['encode']} */
    encode(text, options) {
        return this._encode_plus(text, options).input_ids;
    }

    /** @type {PreTrainedTokenizer['tokenize']} */
    tokenize(text, options) {
        const [language, remainder] = this._split_language_code(text);
        const tokens = super.tokenize(remainder, options);
        return language === null ? tokens : [language, ...tokens];
    }
}
