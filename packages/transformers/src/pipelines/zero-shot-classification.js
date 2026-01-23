import { Pipeline } from './_base.js';

import { softmax } from '../utils/maths.js';

/**
 * @typedef {import('./_base.js').TextPipelineConstructorArgs} TextPipelineConstructorArgs
 * @typedef {import('./_base.js').Disposable} Disposable
 */

/**
 * @typedef {Object} ZeroShotClassificationOutput
 * @property {string} sequence The sequence for which this is the output.
 * @property {string[]} labels The labels sorted by order of likelihood.
 * @property {number[]} scores The probabilities for each of the labels.
 *
 * @typedef {Object} ZeroShotClassificationPipelineOptions Parameters specific to zero-shot classification pipelines.
 * @property {string} [hypothesis_template="This example is {}."] The template used to turn each
 * candidate label into an NLI-style hypothesis. The candidate label will replace the {} placeholder.
 * @property {boolean} [multi_label=false] Whether or not multiple candidate labels can be true.
 * If `false`, the scores are normalized such that the sum of the label likelihoods for each sequence
 * is 1. If `true`, the labels are considered independent and probabilities are normalized for each
 * candidate by doing a softmax of the entailment score vs. the contradiction score.
 *
 * @callback ZeroShotClassificationPipelineCallbackSingle Classify the sequence(s) given as inputs.
 * @param {string} texts The sequence(s) to classify, will be truncated if the model input is too large.
 * @param {string|string[]} candidate_labels The set of possible class labels to classify each sequence into.
 * Can be a single label, a string of comma-separated labels, or a list of labels.
 * @param {ZeroShotClassificationPipelineOptions} [options] The options to use for zero-shot classification.
 * @returns {Promise<ZeroShotClassificationOutput>} An array or object containing the predicted labels and scores.
 *
 * @callback ZeroShotClassificationPipelineCallbackBatch Classify the sequence(s) given as inputs.
 * @param {string[]} texts The sequence(s) to classify, will be truncated if the model input is too large.
 * @param {string|string[]} candidate_labels The set of possible class labels to classify each sequence into.
 * Can be a single label, a string of comma-separated labels, or a list of labels.
 * @param {ZeroShotClassificationPipelineOptions} [options] The options to use for zero-shot classification.
 * @returns {Promise<ZeroShotClassificationOutput[]>} An array or object containing the predicted labels and scores.
 *
 * @typedef {ZeroShotClassificationPipelineCallbackSingle & ZeroShotClassificationPipelineCallbackBatch} ZeroShotClassificationPipelineCallback
 *
 * @typedef {TextPipelineConstructorArgs & ZeroShotClassificationPipelineCallback & Disposable} ZeroShotClassificationPipelineType
 */

/**
 * NLI-based zero-shot classification pipeline using a `ModelForSequenceClassification`
 * trained on NLI (natural language inference) tasks. Equivalent of `text-classification`
 * pipelines, but these models don't require a hardcoded number of potential classes, they
 * can be chosen at runtime. It usually means it's slower but it is **much** more flexible.
 *
 * **Example:** Zero shot classification with `Xenova/mobilebert-uncased-mnli`.
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const classifier = await pipeline('zero-shot-classification', 'Xenova/mobilebert-uncased-mnli');
 * const text = 'Last week I upgraded my iOS version and ever since then my phone has been overheating whenever I use your app.';
 * const labels = [ 'mobile', 'billing', 'website', 'account access' ];
 * const output = await classifier(text, labels);
 * // {
 * //   sequence: 'Last week I upgraded my iOS version and ever since then my phone has been overheating whenever I use your app.',
 * //   labels: [ 'mobile', 'website', 'billing', 'account access' ],
 * //   scores: [ 0.5562091040482018, 0.1843621307860853, 0.13942646639336376, 0.12000229877234923 ]
 * // }
 * ```
 *
 * **Example:** Zero shot classification with `Xenova/nli-deberta-v3-xsmall` (multi-label).
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const classifier = await pipeline('zero-shot-classification', 'Xenova/nli-deberta-v3-xsmall');
 * const text = 'I have a problem with my iphone that needs to be resolved asap!';
 * const labels = [ 'urgent', 'not urgent', 'phone', 'tablet', 'computer' ];
 * const output = await classifier(text, labels, { multi_label: true });
 * // {
 * //   sequence: 'I have a problem with my iphone that needs to be resolved asap!',
 * //   labels: [ 'urgent', 'phone', 'computer', 'tablet', 'not urgent' ],
 * //   scores: [ 0.9958870956360275, 0.9923963400697035, 0.002333537946160235, 0.0015134138567598765, 0.0010699384208377163 ]
 * // }
 * ```
 */
export class ZeroShotClassificationPipeline
    extends /** @type {new (options: TextPipelineConstructorArgs) => ZeroShotClassificationPipelineType} */ (Pipeline)
{
    /**
     * Create a new ZeroShotClassificationPipeline.
     * @param {TextPipelineConstructorArgs} options An object used to instantiate the pipeline.
     */
    constructor(options) {
        super(options);

        // Use model config to get label2id mapping
        this.label2id = Object.fromEntries(
            Object.entries(/** @type {any} */ (this).model.config.label2id).map(([k, v]) => [k.toLowerCase(), v]),
        );

        this.entailment_id = this.label2id['entailment'];
        if (this.entailment_id === undefined) {
            console.warn("Could not find 'entailment' in label2id mapping. Using 2 as entailment_id.");
            this.entailment_id = 2;
        }

        this.contradiction_id = this.label2id['contradiction'] ?? this.label2id['not_entailment'];
        if (this.contradiction_id === undefined) {
            console.warn("Could not find 'contradiction' in label2id mapping. Using 0 as contradiction_id.");
            this.contradiction_id = 0;
        }
    }

    async _call(texts, candidate_labels, { hypothesis_template = 'This example is {}.', multi_label = false } = {}) {
        const isBatched = Array.isArray(texts);
        if (!isBatched) {
            texts = [/** @type {string} */ (texts)];
        }
        if (!Array.isArray(candidate_labels)) {
            candidate_labels = [candidate_labels];
        }

        // Insert labels into hypothesis template
        const hypotheses = candidate_labels.map((x) => hypothesis_template.replace('{}', x));

        // How to perform the softmax over the logits:
        //  - true:  softmax over the entailment vs. contradiction dim for each label independently
        //  - false: softmax the "entailment" logits over all candidate labels
        const softmaxEach = multi_label || candidate_labels.length === 1;

        /** @type {ZeroShotClassificationOutput[]} */
        const toReturn = [];
        for (const premise of texts) {
            const entails_logits = [];

            for (const hypothesis of hypotheses) {
                const inputs = this.tokenizer(premise, {
                    text_pair: hypothesis,
                    padding: true,
                    truncation: true,
                });
                const outputs = await this.model(inputs);

                if (softmaxEach) {
                    entails_logits.push([
                        outputs.logits.data[this.contradiction_id],
                        outputs.logits.data[this.entailment_id],
                    ]);
                } else {
                    entails_logits.push(outputs.logits.data[this.entailment_id]);
                }
            }

            /** @type {number[]} */
            const scores = softmaxEach ? entails_logits.map((x) => softmax(x)[1]) : softmax(entails_logits);

            // Sort by scores (desc) and return scores with indices
            const scores_sorted = scores.map((x, i) => [x, i]).sort((a, b) => b[0] - a[0]);

            toReturn.push({
                sequence: premise,
                labels: scores_sorted.map((x) => candidate_labels[x[1]]),
                scores: scores_sorted.map((x) => x[0]),
            });
        }
        return isBatched ? toReturn : toReturn[0];
    }
}
