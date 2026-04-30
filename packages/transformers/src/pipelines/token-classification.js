import { Pipeline } from './_base.js';

import { max, softmax } from '../utils/maths.js';

/**
 * @typedef {import('./_base.js').TextPipelineConstructorArgs} TextPipelineConstructorArgs
 * @typedef {import('./_base.js').Disposable} Disposable
 */

/**
 * Strategy for fusing tokens based on the model prediction.
 * - `"none"`: Return raw per-token predictions.
 * - `"simple"`: Group entities using BIO / BIOES tags (see pipeline docs for details).
 * @typedef {"none" | "simple"} AggregationStrategy
 */

/**
 * @typedef {Object} TokenClassificationPipelineOptions
 * @property {string[]} [ignore_labels] A list of labels to ignore.
 * @property {AggregationStrategy} [aggregation_strategy="none"] Token-fusion strategy.
 * When set to anything other than `"none"`, results use `entity_group` instead of `entity`.
 */

/**
 * Single element of a token-classification result, parameterised by the options type `O` so that
 * `entity` vs. `entity_group` is known statically based on `aggregation_strategy`.
 *
 * - Grouped (present when `O["aggregation_strategy"]` is `"simple"`):
 *     `{ word, score, entity_group }`
 * - Raw (the default — when `aggregation_strategy` is missing, `"none"`, or `undefined`):
 *     `{ word, score, entity, index }`
 * - Both variants also carry optional `start` / `end` character offsets.
 *
 * When `O` is the untyped `TokenClassificationPipelineOptions`, the element is the union of both shapes,
 * narrowable via `if ("entity_group" in item)` / `if (item.entity !== undefined)`.
 *
 * @template {TokenClassificationPipelineOptions | undefined} [O=TokenClassificationPipelineOptions]
 * @typedef {_PickElement<O>[]} TokenClassificationOutput
 */

/**
 * @template {TokenClassificationPipelineOptions | undefined} O
 * @typedef {O extends undefined
 *   ? _Raw
 *   : "aggregation_strategy" extends keyof O
 *     ? (O extends { aggregation_strategy?: infer A }
 *       ? ([A] extends ["simple"] ? _Grouped
 *         : [A] extends ["none" | undefined] ? _Raw
 *         : _Raw | _Grouped)
 *       : _Raw)
 *     : _Raw} _PickElement
 */

/**
 * @typedef {{ word: string, score: number, entity: string, index: number, entity_group?: undefined, start?: number, end?: number }} _Raw
 * @typedef {{ word: string, score: number, entity_group: string, entity?: undefined, index?: undefined, start?: number, end?: number }} _Grouped
 */

/**
 * @typedef {TextPipelineConstructorArgs & TokenClassificationPipelineCallback & Disposable} TokenClassificationPipelineType
 *
 * @typedef {<Q extends string | string[], const O extends TokenClassificationPipelineOptions = {}>(texts: Q, options?: O) => Promise<Q extends string[] ? TokenClassificationOutput<O>[] : TokenClassificationOutput<O>>} TokenClassificationPipelineCallback
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
 *
 * **Example:** Group adjacent BIO/BIOES tokens into entity spans using `aggregation_strategy: "simple"`.
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const classifier = await pipeline('token-classification', 'Xenova/bert-base-NER');
 * const output = await classifier('My name is Sarah and I live in London', { aggregation_strategy: 'simple' });
 * // [
 * //   { entity_group: 'PER', score: 0.9985477924346924, word: 'Sarah' },
 * //   { entity_group: 'LOC', score: 0.999621570110321, word: 'London' }
 * // ]
 * ```
 */
export class TokenClassificationPipeline
    extends /** @type {new (options: TextPipelineConstructorArgs) => TokenClassificationPipelineType} */ (Pipeline)
{
    async _call(texts, { ignore_labels = ['O'], aggregation_strategy = 'none' } = {}) {
        if (aggregation_strategy !== 'none' && aggregation_strategy !== 'simple') {
            throw new Error(
                `Invalid aggregation_strategy: "${aggregation_strategy}". Must be one of "none" or "simple".`,
            );
        }

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
            const ids = model_inputs.input_ids[i].tolist();
            const batch = logits[i];

            const tokens = [];
            for (let j = 0; j < batch.dims[0]; ++j) {
                const tokenData = batch[j];
                const topScoreIndex = max(tokenData.data)[1];

                const entity = id2label ? id2label[topScoreIndex] : `LABEL_${topScoreIndex}`;
                if (ignore_labels.includes(entity)) continue;

                // TODO add option to keep special tokens?
                const word = this.tokenizer.decode([ids[j]], { skip_special_tokens: true });
                if (word === '') continue; // Was a special token.

                const scores = softmax(tokenData.data);
                tokens.push({
                    entity,
                    score: scores[topScoreIndex],
                    index: j,
                    word,
                    // TODO: Add support for start and end
                });
            }

            toReturn.push(aggregation_strategy === 'simple' ? groupEntities(tokens, ids, this.tokenizer) : tokens);
        }
        return isBatched ? toReturn : toReturn[0];
    }
}

/**
 * Split a raw entity label into its BIOES prefix and tag.
 *
 * @param {string} entity
 * @returns {readonly [prefix: 'B'|'I'|'E'|'S', tag: string]}
 */
function getTag(entity) {
    const p = entity[0];
    return entity[1] === '-' && (p === 'B' || p === 'I' || p === 'E' || p === 'S')
        ? [p, entity.slice(2)]
        : ['I', entity];
}

/**
 * Group raw per-token predictions into entity spans using the SIMPLE strategy.
 *
 * The only "continue" predicate is: a non-`B`/non-`S` token whose tag matches
 * the open group's tag, when that group hasn't been closed by an `E` / `S`.
 * Everything else starts a fresh group.
 *
 * @param {_Raw[]} tokens
 * @param {number[]} ids Full input_ids for the sequence (indexed by `token.index`), used to re-decode
 *   each group so the joined `word` matches what the tokenizer would produce.
 * @param {any} tokenizer
 * @returns {_Grouped[]}
 */
function groupEntities(tokens, ids, tokenizer) {
    /** @type {{ tag: string, start: number, end: number }[]} */
    const groups = []; // each entry is a [start, end) slice into `tokens`, plus the shared tag
    let openTag = null; // null = no open group

    for (let i = 0; i < tokens.length; ++i) {
        const [prefix, tag] = getTag(tokens[i].entity);
        const extend = openTag === tag && prefix !== 'B' && prefix !== 'S';
        if (extend) {
            groups[groups.length - 1].end = i + 1;
            // `E` terminates the group; subsequent `I`/`E`/`S` start fresh.
            if (prefix === 'E') openTag = null;
        } else {
            groups.push({ tag, start: i, end: i + 1 });
            // `S` opens and immediately closes; anything else leaves the group open
            // (including a leading `E` — best-effort recovery for a malformed sequence).
            openTag = prefix === 'S' ? null : tag;
        }
    }

    return groups.map(({ tag, start, end }) => {
        let scoreSum = 0;
        const groupIds = [];
        for (let i = start; i < end; ++i) {
            scoreSum += tokens[i].score;
            groupIds.push(ids[tokens[i].index]);
        }
        return {
            entity_group: tag,
            score: scoreSum / (end - start),
            word: tokenizer.decode(groupIds, { skip_special_tokens: true }),
        };
    });
}
