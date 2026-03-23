import { Pipeline } from './_base.js';

import { max, softmax } from '../utils/maths.js';

/**
 * @typedef {import('./_base.js').TextPipelineConstructorArgs} TextPipelineConstructorArgs
 * @typedef {import('./_base.js').Disposable} Disposable
 */

/**
 * @typedef {Object} TokenClassificationSingle
 * @property {string} word The token/word classified. This is obtained by decoding the selected tokens.
 * @property {number} score The corresponding probability for `entity`.
 * @property {string} entity The entity predicted for that token/word.
 * @property {number} index The index of the corresponding token in the sentence.
 * @property {number} [start] The index of the start of the corresponding entity in the sentence.
 * @property {number} [end] The index of the end of the corresponding entity in the sentence.
 * @typedef {TokenClassificationSingle[]} TokenClassificationOutput
 *
 * @typedef {Object} TokenClassificationPipelineOptions Parameters specific to token classification pipelines.
 * @property {string[]} [ignore_labels] A list of labels to ignore.
 *
 * @callback TokenClassificationPipelineCallbackSingle Classify each token of a single text.
 * @param {string} texts The text to classify.
 * @param {TokenClassificationPipelineOptions} [options] The options to use for token classification.
 * @returns {Promise<TokenClassificationOutput>} The result.
 *
 * @callback TokenClassificationPipelineCallbackBatch Classify each token of multiple texts.
 * @param {string[]} texts The texts to classify.
 * @param {TokenClassificationPipelineOptions} [options] The options to use for token classification.
 * @returns {Promise<TokenClassificationOutput[]>} The result.
 *
 * @typedef {TokenClassificationPipelineCallbackSingle & TokenClassificationPipelineCallbackBatch} TokenClassificationPipelineCallback
 *
 * @typedef {TextPipelineConstructorArgs & TokenClassificationPipelineCallback & Disposable} TokenClassificationPipelineType
 */

/**
 * Named Entity Recognition pipeline using any `ModelForTokenClassification`.
 *
 * **Example:** Perform named entity recognition with `Xenova/bert-base-NER`.
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const classifier = await pipeline('token-classification', 'Xenova/bert-base-NER');
 * const output = await classifier('My name is Sarah and I live in London');
 * // [
 * //   { entity: 'B-PER', score: 0.9980202913284302, index: 4, word: 'Sarah' },
 * //   { entity: 'B-LOC', score: 0.9994474053382874, index: 9, word: 'London' }
 * // ]
 * ```
 *
 * **Example:** Perform named entity recognition with `Xenova/bert-base-NER` (and return all labels).
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const classifier = await pipeline('token-classification', 'Xenova/bert-base-NER');
 * const output = await classifier('Sarah lives in the United States of America', { ignore_labels: [] });
 * // [
 * //   { entity: 'B-PER', score: 0.9966587424278259, index: 1, word: 'Sarah' },
 * //   { entity: 'O', score: 0.9987385869026184, index: 2, word: 'lives' },
 * //   { entity: 'O', score: 0.9990072846412659, index: 3, word: 'in' },
 * //   { entity: 'O', score: 0.9988298416137695, index: 4, word: 'the' },
 * //   { entity: 'B-LOC', score: 0.9995510578155518, index: 5, word: 'United' },
 * //   { entity: 'I-LOC', score: 0.9990395307540894, index: 6, word: 'States' },
 * //   { entity: 'I-LOC', score: 0.9986724853515625, index: 7, word: 'of' },
 * //   { entity: 'I-LOC', score: 0.9975294470787048, index: 8, word: 'America' }
 * // ]
 * ```
 */
export class TokenClassificationPipeline
    extends /** @type {new (options: TextPipelineConstructorArgs) => TokenClassificationPipelineType} */ (Pipeline)
{
    async _call(texts, { ignore_labels = ['O'] } = {}) {
        const isBatched = Array.isArray(texts);

        // Run tokenization
        const model_inputs = this.tokenizer(isBatched ? texts : [texts], {
            padding: true,
            truncation: true,
        });

        // Run model
        const outputs = await this.model(model_inputs);

        const logits = outputs.logits;
        // @ts-expect-error TS2339
        const id2label = this.model.config.id2label;

        const toReturn = [];
        for (let i = 0; i < logits.dims[0]; ++i) {
            const ids = model_inputs.input_ids[i];
            const batch = logits[i];

            // List of tokens that aren't ignored
            const tokens = [];
            for (let j = 0; j < batch.dims[0]; ++j) {
                const tokenData = batch[j];
                const topScoreIndex = max(tokenData.data)[1];

                const entity = id2label ? id2label[topScoreIndex] : `LABEL_${topScoreIndex}`;
                if (ignore_labels.includes(entity)) {
                    // We predicted a token that should be ignored. So, we skip it.
                    continue;
                }

                // TODO add option to keep special tokens?
                const word = this.tokenizer.decode([ids[j].item()], { skip_special_tokens: true });
                if (word === '') {
                    // Was a special token. So, we skip it.
                    continue;
                }

                const scores = softmax(tokenData.data);

                tokens.push({
                    entity: entity,
                    score: scores[topScoreIndex],
                    index: j,
                    word: word,

                    // TODO: Add support for start and end
                    // start: null,
                    // end: null,
                });
            }
            toReturn.push(tokens);
        }
        return isBatched ? toReturn : toReturn[0];
    }
}
