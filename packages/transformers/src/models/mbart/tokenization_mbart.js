import { PreTrainedTokenizer, _build_translation_inputs } from '../../tokenization_utils.js';

export class MBartTokenizer extends PreTrainedTokenizer {
    constructor(tokenizerJSON, tokenizerConfig) {
        super(tokenizerJSON, tokenizerConfig);

        this.languageRegex = /^[a-z]{2}_[A-Z]{2}$/;
        this.language_codes = this.all_special_tokens.filter((x) => this.languageRegex.test(x)).map((x) => x);
        this.lang_to_token = (x) => x; // Identity function
    }

    /**
     * Helper function to build translation inputs for an `MBartTokenizer`.
     * @param {string|string[]} raw_inputs The text to tokenize.
     * @param {Object} tokenizer_options Options to be sent to the tokenizer
     * @param {Object} generate_kwargs Generation options.
     * @returns {Object} Object to be passed to the model.
     */
    _build_translation_inputs(raw_inputs, tokenizer_options, generate_kwargs) {
        return _build_translation_inputs(this, raw_inputs, tokenizer_options, generate_kwargs);
    }
}
