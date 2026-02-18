import { PreTrainedTokenizer, _build_translation_inputs } from '../../tokenization_utils.js';

/**
 * The M2M100Tokenizer class is used to tokenize text for M2M100 ("Many-to-Many") models.
 *
 * M2M100 is a multilingual encoder-decoder (seq-to-seq) model trained for Many-to-Many
 * multilingual translation. It was introduced in this [paper](https://huggingface.co/papers/2010.11125)
 * and first released in [this](https://github.com/pytorch/fairseq/tree/master/examples/m2m_100) repository.
 *
 * For a list of supported languages (along with their language codes),
 * @see {@link https://huggingface.co/facebook/m2m100_418M#languages-covered}
 */
export class M2M100Tokenizer extends PreTrainedTokenizer {
    constructor(tokenizerJSON, tokenizerConfig) {
        super(tokenizerJSON, tokenizerConfig);

        this.languageRegex = /^__[a-z]{2,3}__$/;
        this.language_codes = this.all_special_tokens
            .filter((x) => this.languageRegex.test(x))
            .map((x) => x.slice(2, -2));
        this.lang_to_token = (x) => `__${x}__`;
    }

    /**
     * Helper function to build translation inputs for an `M2M100Tokenizer`.
     * @param {string|string[]} raw_inputs The text to tokenize.
     * @param {Object} tokenizer_options Options to be sent to the tokenizer
     * @param {Object} generate_kwargs Generation options.
     * @returns {Object} Object to be passed to the model.
     */
    _build_translation_inputs(raw_inputs, tokenizer_options, generate_kwargs) {
        return _build_translation_inputs(this, raw_inputs, tokenizer_options, generate_kwargs);
    }
}
