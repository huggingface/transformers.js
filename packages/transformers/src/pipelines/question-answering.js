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
 * @property {number} [start] The character start index of the answer (in the tokenized version of the input).
 * @property {number} [end] The character end index of the answer (in the tokenized version of the input).
 * @property {string} answer The answer to the question.
 *
 * @typedef {Object} QuestionAnsweringPipelineOptions Parameters specific to question answering pipelines.
 * @property {number} [top_k=1] The number of top answer predictions to be returned.
 *
 * @callback QuestionAnsweringPipelineCallback Answer the question(s) given as inputs by using the context(s).
 * @param {string|string[]} question One or several question(s) (must be used in conjunction with the `context` argument).
 * @param {string|string[]} context One or several context(s) associated with the question(s) (must be used in conjunction with the `question` argument).
 * @param {QuestionAnsweringPipelineOptions} [options] The options to use for question answering.
 * @returns {Promise<QuestionAnsweringOutput|QuestionAnsweringOutput[]>} An array or object containing the predicted answers and scores.
 *
 * @typedef {TextPipelineConstructorArgs & QuestionAnsweringPipelineCallback & Disposable} QuestionAnsweringPipelineType
 */

/**
 * Question Answering pipeline using any `ModelForQuestionAnswering`.
 *
 * **Example:** Run question answering with `Xenova/distilbert-base-uncased-distilled-squad`.
 * ```javascript
 * const answerer = await pipeline('question-answering', 'Xenova/distilbert-base-uncased-distilled-squad');
 * const question = 'Who was Jim Henson?';
 * const context = 'Jim Henson was a nice puppet.';
 * const output = await answerer(question, context);
 * // {
 * //   answer: "a nice puppet",
 * //   score: 0.5768911502526741
 * // }
 * ```
 */
export class QuestionAnsweringPipeline
    extends /** @type {new (options: TextPipelineConstructorArgs) => QuestionAnsweringPipelineType} */ (Pipeline)
{
    /**
     * Create a new QuestionAnsweringPipeline.
     * @param {TextPipelineConstructorArgs} options An object used to instantiate the pipeline.
     */
    constructor(options) {
        super(options);
    }

    /** @type {QuestionAnsweringPipelineCallback} */
    async _call(question, context, { top_k = 1 } = {}) {
        // Run tokenization
        const inputs = this.tokenizer(question, {
            text_pair: context,
            padding: true,
            truncation: true,
        });

        const { start_logits, end_logits } = await this.model(inputs);
        const input_ids = inputs.input_ids.tolist();
        const attention_mask = inputs.attention_mask.tolist();

        // TODO: add support for `return_special_tokens_mask`
        const special_tokens = this.tokenizer.all_special_ids;

        /** @type {QuestionAnsweringOutput[]} */
        const toReturn = [];
        for (let j = 0; j < start_logits.dims[0]; ++j) {
            const ids = input_ids[j];
            const sepIndex = ids.findIndex(
                (x) =>
                    // We use == to match bigint with number
                    // @ts-ignore
                    x == this.tokenizer.sep_token_id,
            );

            const valid_mask = attention_mask[j].map(
                (y, ix) =>
                    y == 1 &&
                    (ix === 0 || // is cls_token
                        (ix > sepIndex && special_tokens.findIndex((x) => x == ids[ix]) === -1)), // token is not a special token (special_tokens_mask == 0)
            );

            const start = start_logits[j].tolist();
            const end = end_logits[j].tolist();

            // Now, we mask out values that can't be in the answer
            // NOTE: We keep the cls_token unmasked (some models use it to indicate unanswerable questions)
            for (let i = 1; i < start.length; ++i) {
                if (
                    attention_mask[j] == 0 || // is part of padding
                    i <= sepIndex || // is before the sep_token
                    special_tokens.findIndex((x) => x == ids[i]) !== -1 // Is a special token
                ) {
                    // Make sure non-context indexes in the tensor cannot contribute to the softmax
                    start[i] = -Infinity;
                    end[i] = -Infinity;
                }
            }

            // Normalize logits and spans to retrieve the answer
            const start_scores = softmax(start).map((x, i) => [x, i]);
            const end_scores = softmax(end).map((x, i) => [x, i]);

            // Mask CLS
            start_scores[0][0] = 0;
            end_scores[0][0] = 0;

            // Generate all valid spans and select best ones
            const options = product(start_scores, end_scores)
                .filter((x) => x[0][1] <= x[1][1])
                .map((x) => [x[0][1], x[1][1], x[0][0] * x[1][0]])
                .sort((a, b) => b[2] - a[2]);

            for (let k = 0; k < Math.min(options.length, top_k); ++k) {
                const [start, end, score] = options[k];

                const answer_tokens = ids.slice(start, end + 1);

                const answer = this.tokenizer.decode(answer_tokens, {
                    skip_special_tokens: true,
                });

                // TODO add start and end?
                // NOTE: HF returns character index
                toReturn.push({
                    answer,
                    score,
                });
            }
        }

        // Mimic HF's return type based on top_k
        return top_k === 1 ? toReturn[0] : toReturn;
    }
}
