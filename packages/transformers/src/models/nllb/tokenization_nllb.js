import { PreTrainedTokenizer, _build_translation_inputs } from '../../tokenization_utils.js';

/**
 * The NllbTokenizer class is used to tokenize text for NLLB ("No Language Left Behind") models.
 *
 * No Language Left Behind (NLLB) is a first-of-its-kind, AI breakthrough project
 * that open-sources models capable of delivering high-quality translations directly
 * between any pair of 200+ languages — including low-resource languages like Asturian,
 * Luganda, Urdu and more. It aims to help people communicate with anyone, anywhere,
 * regardless of their language preferences. For more information, check out their
 * [paper](https://huggingface.co/papers/2207.04672).
 *
 * For a list of supported languages (along with their language codes),
 * @see {@link https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200}
 */
export class NllbTokenizer extends PreTrainedTokenizer {
    constructor(tokenizerJSON, tokenizerConfig) {
        super(tokenizerJSON, tokenizerConfig);

        this.languageRegex = /^[a-z]{3}_[A-Z][a-z]{3}$/;
        this.language_codes = this.all_special_tokens.filter((x) => this.languageRegex.test(x));
        this.lang_to_token = (x) => x; // Identity function
    }

    /**
     * Helper function to build translation inputs for an `NllbTokenizer`.
     * @param {string|string[]} raw_inputs The text to tokenize.
     * @param {Object} tokenizer_options Options to be sent to the tokenizer
     * @param {Object} generate_kwargs Generation options.
     * @returns {Object} Object to be passed to the model.
     */
    _build_translation_inputs(raw_inputs, tokenizer_options, generate_kwargs) {
        return _build_translation_inputs(this, raw_inputs, tokenizer_options, generate_kwargs);
    }
}
