import { Pipeline } from './_base.js';

import { product } from '../utils/core.js';
import { softmax } from '../utils/maths.js';

/**
 * @typedef {import('./_base.js').TextPipelineConstructorArgs} TextPipelineConstructorArgs
 * @typedef {import('./_base.js').Disposable} Disposable
 */

/**
 * @typedef {Object} QuestionAnsweringOutput
 * @property {number} score The probability associated to the answer.
 * @property {number} start The answer start offset (character index **in `context`**; slice with `context.slice(start, end)`).
 * @property {number} end The exclusive end offset of the answer in **`context`** (half-open `[start, end)`).
 * @property {string} answer The answer to the question.
 *
 * @typedef {Object} QuestionAnsweringPipelineOptions Parameters specific to question answering pipelines.
 * @property {number} [top_k=1] The number of top answer predictions to be returned.
 *
 * @typedef {TextPipelineConstructorArgs & QuestionAnsweringPipelineCallback & Disposable} QuestionAnsweringPipelineType
 */

/**
 * @template O
 * @typedef {O extends { top_k: infer K } ? (1 extends K ? false : true) : false} QuestionAnsweringIsTopK
 */

/**
 * @template Q, O
 * @typedef {Q extends string[] ? (QuestionAnsweringIsTopK<O> extends true ? QuestionAnsweringOutput[][] : QuestionAnsweringOutput[]) : (QuestionAnsweringIsTopK<O> extends true ? QuestionAnsweringOutput[] : QuestionAnsweringOutput)} QuestionAnsweringPipelineResult
 */

/**
 * @typedef {<Q extends string | string[], const O extends { top_k?: number } = {}>(question: Q, context: Q, options?: O) => Promise<QuestionAnsweringPipelineResult<Q, O>>} QuestionAnsweringPipelineCallback
 */

/**
 * Question Answering pipeline using any `ModelForQuestionAnswering`.
 *
 * **Example:** Run question answering with `Xenova/distilbert-base-uncased-distilled-squad`.
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const answerer = await pipeline('question-answering', 'Xenova/distilbert-base-uncased-distilled-squad');
 * const question = 'Who was Jim Henson?';
 * const context = 'Jim Henson was a nice puppet.';
 * const output = await answerer(question, context);
 * // {
 * //   answer: "a nice puppet",
 * //   score: 0.5768911502526741,
 * //   start: ...,
 * //   end: ...,
 * // }
 * ```
 */
export class QuestionAnsweringPipeline
    extends /** @type {new (options: TextPipelineConstructorArgs) => QuestionAnsweringPipelineType} */ (Pipeline)
{
    async _call(question, context, { top_k = 1 } = {}) {
        // Run tokenization
        const inputs = this.tokenizer(question, {
            text_pair: context,
            padding: true,
            truncation: true,
            return_offsets_mapping: true,
        });
        const isBatched = Array.isArray(question);

        const { start_logits, end_logits } = await this.model(inputs);
        const input_ids = inputs.input_ids.tolist();
        const attention_mask = inputs.attention_mask.tolist();

        // TODO: add support for `return_special_tokens_mask`
        const { all_special_ids, sep_token_id } = this.tokenizer;

        const offset_mapping_batches = inputs.offset_mapping;
        /** @type {QuestionAnsweringOutput[][]|QuestionAnsweringOutput[]} */
        const batchedResults = [];

        for (let j = 0; j < start_logits.dims[0]; ++j) {
            const ids = input_ids[j];
            const sepIndex = ids.findIndex(
                (x) =>
                    // We use == to match bigint with number
                    // @ts-ignore
                    x == sep_token_id,
            );

            const start_logits_row = start_logits[j].tolist();
            const end_logits_row = end_logits[j].tolist();

            // Now, we mask out values that can't be in the answer
            // NOTE: We keep the cls_token unmasked (some models use it to indicate unanswerable questions)
            for (let i = 1; i < start_logits_row.length; ++i) {
                if (
                    attention_mask[j] == 0 || // is part of padding
                    i <= sepIndex || // is before the sep_token
                    all_special_ids.findIndex((x) => x == ids[i]) !== -1 // Is a special token
                ) {
                    // Make sure non-context indexes in the tensor cannot contribute to the softmax
                    start_logits_row[i] = -Infinity;
                    end_logits_row[i] = -Infinity;
                }
            }

            // Normalize logits and spans to retrieve the answer
            const start_scores = softmax(start_logits_row).map((x, i) => [x, i]);
            const end_scores = softmax(end_logits_row).map((x, i) => [x, i]);

            // Mask CLS
            start_scores[0][0] = 0;
            end_scores[0][0] = 0;

            // Generate all valid spans and select best ones
            const options = product(start_scores, end_scores)
                .filter((x) => x[0][1] <= x[1][1])
                .map((x) => [x[0][1], x[1][1], x[0][0] * x[1][0]])
                .sort((a, b) => b[2] - a[2]);

            const rowOffsets = isBatched ? offset_mapping_batches[j] : offset_mapping_batches;

            const sampleResults = [];
            for (let k = 0; k < Math.min(options.length, top_k); ++k) {
                const [startTok, endTok, spanScore] = options[k];

                const answer_tokens = ids.slice(startTok, endTok + 1);

                const answer = this.tokenizer.decode(answer_tokens, {
                    skip_special_tokens: true,
                });

                /** @type {number} */
                const startChar = rowOffsets[startTok][0];
                /** @type {number} */
                const endChar = rowOffsets[endTok][1];

                sampleResults.push({
                    answer,
                    score: spanScore,
                    start: startChar,
                    end: endChar,
                });
            }
            if (top_k === 1) {
                batchedResults.push(...sampleResults);
            } else {
                batchedResults.push(sampleResults);
            }
        }

        return isBatched ? batchedResults : batchedResults[0];
    }
}
