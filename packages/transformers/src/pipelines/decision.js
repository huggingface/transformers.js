/**
 * @module pipelines
 */

import { Pipeline } from './_base.js';

import { Tensor } from '../utils/tensor.js';
import { softmax } from '../utils/maths.js';
import { DecisionConfig } from '../decision/configuration_utils.js';
import { label_token_ids } from '../decision/template_utils.js';

/**
 * @typedef {import('./_base.js').TextPipelineConstructorArgs} TextPipelineConstructorArgs
 * @typedef {import('./_base.js').Disposable} Disposable
 * @typedef {import('../decision/template_utils.js').DecisionRequest} DecisionRequest
 * @typedef {import('../decision/template_utils.js').DecisionAnswer} DecisionAnswer
 */

/**
 * @typedef {Record<string, DecisionAnswer>} DecisionOutput The answers to a request, by question name.
 *
 * @typedef {Object} DecisionPipelineConstructorArgs
 * @property {DecisionConfig} [decision_config] The decision config. Defaults to the one of the model repo.
 *
 * @typedef {TextPipelineConstructorArgs & DecisionPipelineConstructorArgs} DecisionPipelineArgs
 *
 * @typedef {TextPipelineConstructorArgs & DecisionPipelineCallback & Disposable} DecisionPipelineType
 */

/**
 * @template T
 * @typedef {T extends Array<any> ? DecisionOutput[] : DecisionOutput} DecisionPipelineResult
 */

/**
 * @typedef {<T extends DecisionRequest | string | Array<DecisionRequest | string>>(requests: T) => Promise<DecisionPipelineResult<T>>} DecisionPipelineCallback
 */

/**
 * Decision pipeline (experimental, ported from `transformers`): answers typed questions (`choice`, `score`, `noul`)
 * about a `state` with a probability for every option, in the System One format. Nothing is generated: the score of
 * each option is the next-token logit of its label (`A`, `B`, ... or `Yes`/`No`), so any chat model can answer. The
 * prompt and labels come from the `decision.json` and `decision_template.jinja` of the model repo, if any.
 *
 * **Example:** Typed decisions with `onnx-community/Qwen2.5-0.5B-Instruct`.
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const decide = await pipeline('decision', 'onnx-community/Qwen2.5-0.5B-Instruct');
 * const answers = await decide({
 *   state: 'I was charged twice for my order and want a refund.',
 *   questions: {
 *     intent: {
 *       type: 'choice',
 *       instructions: 'What does the customer want?',
 *       criteria: { refund: 'wants money back', cancel: 'wants to cancel an order', track: 'wants to track an order' },
 *     },
 *     refund: { type: 'noul', instructions: 'Is a refund requested?' },
 *     anger: { type: 'score', instructions: 'How angry?', criteria: ['calm', 'annoyed', 'furious'] },
 *   },
 * });
 * // {
 * //   intent: { type: 'choice', choice: 'refund', confidence: ..., probabilities: { refund: ..., cancel: ..., track: ... } },
 * //   refund: { type: 'noul', noul: ... },
 * //   anger: { type: 'score', score: ..., legend: { 0: 'calm', 1: 'annoyed', 2: 'furious' }, confidence: ..., probabilities: { 0: ..., 1: ..., 2: ... } },
 * // }
 * ```
 */
export class DecisionPipeline
    extends /** @type {new (options: DecisionPipelineArgs) => DecisionPipelineType} */ (Pipeline)
{
    /**
     * Create a new DecisionPipeline.
     * @param {DecisionPipelineArgs} options An object used to instantiate the pipeline.
     */
    constructor(options) {
        super(options);
        this.decision_config = options.decision_config ?? new DecisionConfig();
        if (this.decision_config.adapter) {
            throw new Error(
                "LoRA adapters (`adapter` in the decision config) aren't supported: merge them into the model.",
            );
        }
        /** @type {Map<string, number[]>} */
        this.label_ids = new Map();
    }

    /**
     * @param {DecisionRequest|string} request
     * @returns {Promise<DecisionOutput>}
     * @private
     */
    async _decide(request) {
        /** @type {DecisionOutput} */
        const answers = {};
        for (const { input_ids, marks, questions } of this.decision_config.prepare_inputs(request, this.tokenizer)) {
            const length = input_ids.length;
            const { logits } = await this.model({
                input_ids: new Tensor('int64', BigInt64Array.from(input_ids.map(BigInt)), [1, length]),
                attention_mask: new Tensor('int64', new BigInt64Array(length).fill(1n), [1, length]),
            });
            const vocab_size = logits.dims.at(-1);
            questions.forEach((question, i) => {
                // The logits of the labels at the token before each mark
                const key = JSON.stringify(question.labels);
                if (!this.label_ids.has(key)) this.label_ids.set(key, label_token_ids(this.tokenizer, question.labels));
                const offset = (marks[i] - 1) * vocab_size;
                const temperature = this.decision_config.get_temperature(question.type);
                const scores = this.label_ids.get(key).map((id) => logits.data[offset + id] / temperature);
                answers[question.name] = question.answer(softmax(scores));
            });
        }
        return answers;
    }

    /** @type {DecisionPipelineCallback} */
    async _call(requests) {
        if (!Array.isArray(requests)) return /** @type {any} */ (await this._decide(requests));
        const outputs = [];
        for (const request of requests) {
            outputs.push(await this._decide(request));
        }
        return /** @type {any} */ (outputs);
    }
}
