import { Text2TextGenerationPipeline } from './text2text-generation.js';

/**
 * @typedef {import('./_base.js').TextPipelineConstructorArgs} TextPipelineConstructorArgs
 * @typedef {import('./_base.js').Disposable} Disposable
 */

/**
 * @typedef {Object} TranslationSingle
 * @property {string} translation_text The translated text.
 * @typedef {TranslationSingle[]} TranslationOutput
 *
 * @callback TranslationPipelineCallback Translate the text(s) given as inputs.
 * @param {string|string[]} texts Texts to be translated.
 * @param {import('./generation/configuration_utils.js').GenerationConfig} [options] Additional keyword arguments to pass along to the generate method of the model.
 * @returns {Promise<TranslationOutput>}
 *
 * @typedef {TextPipelineConstructorArgs & TranslationPipelineCallback & Disposable} TranslationPipelineType
 */

/**
 * Translates text from one language to another.
 *
 * **Example:** Multilingual translation w/ `Xenova/nllb-200-distilled-600M`.
 *
 * See [here](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200)
 * for the full list of languages and their corresponding codes.
 *
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const translator = await pipeline('translation', 'Xenova/nllb-200-distilled-600M');
 * const output = await translator('जीवन एक चॉकलेट बॉक्स की तरह है।', {
 *   src_lang: 'hin_Deva', // Hindi
 *   tgt_lang: 'fra_Latn', // French
 * });
 * // [{ translation_text: 'La vie est comme une boîte à chocolat.' }]
 * ```
 *
 * **Example:** Multilingual translation w/ `Xenova/m2m100_418M`.
 *
 * See [here](https://huggingface.co/facebook/m2m100_418M#languages-covered)
 * for the full list of languages and their corresponding codes.
 *
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const translator = await pipeline('translation', 'Xenova/m2m100_418M');
 * const output = await translator('生活就像一盒巧克力。', {
 *   src_lang: 'zh', // Chinese
 *   tgt_lang: 'en', // English
 * });
 * // [{ translation_text: 'Life is like a box of chocolate.' }]
 * ```
 *
 * **Example:** Multilingual translation w/ `Xenova/mbart-large-50-many-to-many-mmt`.
 *
 * See [here](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt#languages-covered)
 * for the full list of languages and their corresponding codes.
 *
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const translator = await pipeline('translation', 'Xenova/mbart-large-50-many-to-many-mmt');
 * const output = await translator('संयुक्त राष्ट्र के प्रमुख का कहना है कि सीरिया में कोई सैन्य समाधान नहीं है', {
 *   src_lang: 'hi_IN', // Hindi
 *   tgt_lang: 'fr_XX', // French
 * });
 * // [{ translation_text: 'Le chef des Nations affirme qu 'il n 'y a military solution in Syria.' }]
 * ```
 */
export class TranslationPipeline
    extends /** @type {new (options: TextPipelineConstructorArgs) => TranslationPipelineType} */ (
        /** @type {any} */ (Text2TextGenerationPipeline)
    )
{
    /** @type {'translation_text'} */
    _key = 'translation_text';

    /**
     * Create a new TranslationPipeline.
     * @param {TextPipelineConstructorArgs} options An object used to instantiate the pipeline.
     */
    constructor(options) {
        super(options);
    }
}
