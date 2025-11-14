import { Pipeline } from './_base.js';

import { Tensor, topk } from '../utils/tensor.js';
import { softmax } from '../utils/maths.js';

/**
 * @typedef {import('./_base.js').TextPipelineConstructorArgs} TextPipelineConstructorArgs
 * @typedef {import('./_base.js').Disposable} Disposable
 */

/**
 * @typedef {Object} FillMaskSingle
 * @property {string} sequence The corresponding input with the mask token prediction.
 * @property {number} score The corresponding probability.
 * @property {number} token The predicted token id (to replace the masked one).
 * @property {string} token_str The predicted token (to replace the masked one).
 * @typedef {FillMaskSingle[]} FillMaskOutput
 *
 * @typedef {Object} FillMaskPipelineOptions Parameters specific to fill mask pipelines.
 * @property {number} [top_k=5] When passed, overrides the number of predictions to return.
 *
 * @callback FillMaskPipelineCallback Fill the masked token in the text(s) given as inputs.
 * @param {string|string[]} texts One or several texts (or one list of prompts) with masked tokens.
 * @param {FillMaskPipelineOptions} [options] The options to use for masked language modelling.
 * @returns {Promise<FillMaskOutput|FillMaskOutput[]>} An array of objects containing the score, predicted token, predicted token string,
 * and the sequence with the predicted token filled in, or an array of such arrays (one for each input text).
 * If only one input text is given, the output will be an array of objects.
 * @throws {Error} When the mask token is not found in the input text.
 *
 * @typedef {TextPipelineConstructorArgs & FillMaskPipelineCallback & Disposable} FillMaskPipelineType
 */

/**
 * Masked language modeling prediction pipeline using any `ModelWithLMHead`.
 *
 * **Example:** Perform masked language modelling (a.k.a. "fill-mask") with `Xenova/bert-base-uncased`.
 * ```javascript
 * const unmasker = await pipeline('fill-mask', 'Xenova/bert-base-cased');
 * const output = await unmasker('The goal of life is [MASK].');
 * // [
 * //   { token_str: 'survival', score: 0.06137419492006302, token: 8115, sequence: 'The goal of life is survival.' },
 * //   { token_str: 'love', score: 0.03902450203895569, token: 1567, sequence: 'The goal of life is love.' },
 * //   { token_str: 'happiness', score: 0.03253183513879776, token: 9266, sequence: 'The goal of life is happiness.' },
 * //   { token_str: 'freedom', score: 0.018736306577920914, token: 4438, sequence: 'The goal of life is freedom.' },
 * //   { token_str: 'life', score: 0.01859794743359089, token: 1297, sequence: 'The goal of life is life.' }
 * // ]
 * ```
 *
 * **Example:** Perform masked language modelling (a.k.a. "fill-mask") with `Xenova/bert-base-cased` (and return top result).
 * ```javascript
 * const unmasker = await pipeline('fill-mask', 'Xenova/bert-base-cased');
 * const output = await unmasker('The Milky Way is a [MASK] galaxy.', { top_k: 1 });
 * // [{ token_str: 'spiral', score: 0.6299987435340881, token: 14061, sequence: 'The Milky Way is a spiral galaxy.' }]
 * ```
 */
export class FillMaskPipeline
    extends /** @type {new (options: TextPipelineConstructorArgs) => FillMaskPipelineType} */ (Pipeline)
{
    /**
     * Create a new FillMaskPipeline.
     * @param {TextPipelineConstructorArgs} options An object used to instantiate the pipeline.
     */
    constructor(options) {
        super(options);
    }

    /** @type {FillMaskPipelineCallback} */
    async _call(texts, { top_k = 5 } = {}) {
        // Run tokenization
        const model_inputs = this.tokenizer(texts, {
            padding: true,
            truncation: true,
        });

        // Run model
        const { logits } = await this.model(model_inputs);

        const toReturn = [];

        /** @type {bigint[][]} */
        const input_ids = model_inputs.input_ids.tolist();
        for (let i = 0; i < input_ids.length; ++i) {
            const ids = input_ids[i];
            const mask_token_index = ids.findIndex(
                (x) =>
                    // We use == to match bigint with number
                    // @ts-ignore
                    x == this.tokenizer.mask_token_id,
            );
            if (mask_token_index === -1) {
                throw Error(`Mask token (${this.tokenizer.mask_token}) not found in text.`);
            }
            const itemLogits = logits[i][mask_token_index];

            const scores = await topk(new Tensor('float32', softmax(itemLogits.data), itemLogits.dims), top_k);
            const values = scores[0].tolist();
            const indices = scores[1].tolist();

            toReturn.push(
                indices.map((x, i) => {
                    const sequence = ids.slice();
                    sequence[mask_token_index] = x;

                    return {
                        score: values[i],
                        token: Number(x),
                        token_str: this.tokenizer.decode([x]),
                        sequence: this.tokenizer.decode(sequence, { skip_special_tokens: true }),
                    };
                }),
            );
        }
        return Array.isArray(texts) ? toReturn : toReturn[0];
    }
}
