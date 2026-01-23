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
 * @callback FillMaskPipelineCallbackSingle Fill the masked token in the text given as input.
 * @param {string} texts The text with masked tokens.
 * @param {FillMaskPipelineOptions} [options] The options to use for masked language modelling.
 * @returns {Promise<FillMaskOutput>} An array of objects containing the score, predicted token, predicted token string,
 * and the sequence with the predicted token filled in.
 *
 * @callback FillMaskPipelineCallbackBatch Fill the masked token in the texts given as inputs.
 * @param {string[]} texts A list of texts with masked tokens.
 * @param {FillMaskPipelineOptions} [options] The options to use for masked language modelling.
 * @returns {Promise<FillMaskOutput[]>} An array where each entry corresponds to the predictions for an input text.
 *
 * @typedef {FillMaskPipelineCallbackSingle & FillMaskPipelineCallbackBatch} FillMaskPipelineCallback
 *
 * @typedef {TextPipelineConstructorArgs & FillMaskPipelineCallback & Disposable} FillMaskPipelineType
 */

/**
 * Masked language modeling prediction pipeline using any `ModelWithLMHead`.
 *
 * **Example:** Perform masked language modelling (a.k.a. "fill-mask") with `onnx-community/ettin-encoder-32m-ONNX`.
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const unmasker = await pipeline('fill-mask', 'onnx-community/ettin-encoder-32m-ONNX');
 * const output = await unmasker('The capital of France is [MASK].');
 * // [
 * //   { score: 0.5151872038841248, token: 7785, token_str: ' Paris', sequence: 'The capital of France is Paris.' },
 * //   { score: 0.033725105226039886, token: 42268, token_str: ' Lyon', sequence: 'The capital of France is Lyon.' },
 * //   { score: 0.031234024092555046, token: 23397, token_str: ' Nancy', sequence: 'The capital of France is Nancy.' },
 * //   { score: 0.02075139433145523, token: 30167, token_str: ' Brussels', sequence: 'The capital of France is Brussels.' },
 * //   { score: 0.018962178379297256, token: 31955, token_str: ' Geneva', sequence: 'The capital of France is Geneva.' }
 * // ]
 * ```
 *
 * **Example:** Perform masked language modelling (a.k.a. "fill-mask") with `Xenova/bert-base-uncased`.
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const unmasker = await pipeline('fill-mask', 'Xenova/bert-base-cased');
 * const output = await unmasker('The goal of life is [MASK].');
 * // [
 * //   { score: 0.11368396878242493, sequence: "The goal of life is survival.", token: 8115, token_str: "survival" },
 * //   { score: 0.053510840982198715, sequence: "The goal of life is love.", token: 1567, token_str: "love" },
 * //   { score: 0.05041185021400452, sequence: "The goal of life is happiness.", token: 9266, token_str: "happiness" },
 * //   { score: 0.033218126744031906, sequence: "The goal of life is freedom.", token: 4438, token_str: "freedom" },
 * //   { score: 0.03301157429814339, sequence: "The goal of life is success.", token: 2244, token_str: "success" },
 * // ]
 * ```
 *
 * **Example:** Perform masked language modelling (a.k.a. "fill-mask") with `Xenova/bert-base-cased` (and return top result).
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const unmasker = await pipeline('fill-mask', 'Xenova/bert-base-cased');
 * const output = await unmasker('The Milky Way is a [MASK] galaxy.', { top_k: 1 });
 * // [{ score: 0.5982972383499146, sequence: "The Milky Way is a spiral galaxy.", token: 14061, token_str: "spiral" }]
 * ```
 */
export class FillMaskPipeline
    extends /** @type {new (options: TextPipelineConstructorArgs) => FillMaskPipelineType} */ (Pipeline)
{
    async _call(texts, { top_k = 5 } = {}) {
        const { mask_token_id, mask_token } = this.tokenizer;

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
                    // @ts-ignore - TS2367: Intentional loose equality for bigint/number comparison
                    x == mask_token_id,
            );
            if (mask_token_index === -1) {
                throw Error(`Mask token (${mask_token}) not found in text.`);
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
                        sequence: this.tokenizer.decode(sequence, {
                            skip_special_tokens: true,
                        }),
                    };
                }),
            );
        }
        return Array.isArray(texts) ? toReturn : toReturn[0];
    }
}
