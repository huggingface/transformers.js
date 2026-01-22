import { Pipeline } from './_base.js';

import { Tensor } from '../utils/tensor.js';

/**
 * @typedef {import('./_base.js').TextPipelineConstructorArgs} TextPipelineConstructorArgs
 * @typedef {import('./_base.js').Disposable} Disposable
 */

/**
 * @typedef {Object} Text2TextGenerationSingle
 * @property {string} generated_text The generated text.
 * @typedef {Text2TextGenerationSingle[]} Text2TextGenerationOutput
 *
 * @callback Text2TextGenerationPipelineCallback Generate the output text(s) using text(s) given as inputs.
 * @param {string|string[]} texts Input text for the encoder.
 * @param {Partial<import('../generation/configuration_utils.js').GenerationConfig>} [options] Additional keyword arguments to pass along to the generate method of the model.
 * @returns {Promise<Text2TextGenerationOutput>}
 *
 * @typedef {TextPipelineConstructorArgs & Text2TextGenerationPipelineCallback & Disposable} Text2TextGenerationPipelineType
 */

/**
 * Text2TextGenerationPipeline class for generating text using a model that performs text-to-text generation tasks.
 *
 * **Example:** Text-to-text generation w/ `Xenova/LaMini-Flan-T5-783M`.
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const generator = await pipeline('text2text-generation', 'Xenova/LaMini-Flan-T5-783M');
 * const output = await generator('how can I become more healthy?', {
 *   max_new_tokens: 100,
 * });
 * // [{ generated_text: "To become more healthy, you can: 1. Eat a balanced diet with plenty of fruits, vegetables, whole grains, lean proteins, and healthy fats. 2. Stay hydrated by drinking plenty of water. 3. Get enough sleep and manage stress levels. 4. Avoid smoking and excessive alcohol consumption. 5. Regularly exercise and maintain a healthy weight. 6. Practice good hygiene and sanitation. 7. Seek medical attention if you experience any health issues." }]
 * ```
 */
export class Text2TextGenerationPipeline
    extends /** @type {new (options: TextPipelineConstructorArgs) => Text2TextGenerationPipelineType} */ (Pipeline)
{
    /** @type {'generated_text'} */
    _key = 'generated_text';

    /** @type {Text2TextGenerationPipelineCallback} */
    async _call(texts, generate_kwargs = {}) {
        if (!Array.isArray(texts)) {
            texts = [texts];
        }

        // Add global prefix, if present
        // @ts-expect-error TS2339
        if (this.model.config.prefix) {
            // @ts-expect-error TS2339
            texts = texts.map((x) => this.model.config.prefix + x);
        }

        // Handle task specific params:
        // @ts-expect-error TS2339
        const task_specific_params = this.model.config.task_specific_params;
        if (task_specific_params && task_specific_params[this.task]) {
            // Add prefixes, if present
            if (task_specific_params[this.task].prefix) {
                texts = texts.map((x) => task_specific_params[this.task].prefix + x);
            }

            // TODO update generation config
        }

        const tokenizer = this.tokenizer;
        const tokenizer_options = {
            padding: true,
            truncation: true,
        };
        let inputs;
        if (this.task === 'translation' && '_build_translation_inputs' in tokenizer) {
            // TODO: move to Translation pipeline?
            // Currently put here to avoid code duplication
            // @ts-ignore
            inputs = tokenizer._build_translation_inputs(texts, tokenizer_options, generate_kwargs);
        } else {
            inputs = tokenizer(texts, tokenizer_options);
        }

        const outputTokenIds = await this.model.generate({ ...inputs, ...generate_kwargs });
        return tokenizer
            .batch_decode(/** @type {Tensor} */ (outputTokenIds), {
                skip_special_tokens: true,
            })
            .map((text) => ({ [this._key]: text }));
    }
}
