/**
 * @module generation/logits_sampler
 */

import { Callable } from '../utils/generic.js';
import { Tensor, topk } from '../utils/tensor.js';

import { max, softmax } from '../utils/maths.js';
import { _weightedIndex } from '../utils/random.js';
import { GenerationConfig } from '../generation/configuration_utils.js';
import { TopKLogitsWarper, TopPLogitsWarper } from '../generation/logits_process.js';

/**
 * Sampler is a base class for all sampling methods used for text generation.
 */
export class LogitsSampler extends Callable {
    /**
     * Creates a new Sampler object with the specified generation config.
     * @param {GenerationConfig} generation_config The generation config.
     */
    constructor(generation_config) {
        super();
        this.generation_config = generation_config;
    }

    /**
     * Executes the sampler, using the specified logits.
     * @param {Tensor} logits
     * @returns {Promise<[bigint, number][]>}
     */
    async _call(logits) {
        // Sample from logits, of dims [batch, sequence_length, vocab_size].
        // If index is specified, sample from [batch, index, vocab_size].
        return this.sample(logits);
    }

    /**
     * Abstract method for sampling the logits.
     * @param {Tensor} logits
     * @throws {Error} If not implemented in subclass.
     * @returns {Promise<[bigint, number][]>}
     */
    async sample(logits) {
        throw Error('sample should be implemented in subclasses.');
    }

    /**
     * Returns the specified logits as an array, with temperature applied.
     * @param {Tensor} logits
     * @param {number} index
     * @returns {Float32Array}
     */
    getLogits(logits, index) {
        let vocabSize = logits.dims.at(-1);

        let logs = /** @type {Float32Array} */ (logits.data);

        if (index === -1) {
            logs = logs.slice(-vocabSize);
        } else {
            let startIndex = index * vocabSize;
            logs = logs.slice(startIndex, startIndex + vocabSize);
        }
        return logs;
    }

    /**
     * Selects an item randomly based on the specified probabilities.
     * @param {Float32Array} probabilities An array of probabilities to use for selection.
     * @returns {number} The index of the selected item.
     */
    randomSelect(probabilities) {
        return _weightedIndex(probabilities);
    }

    /**
     * Returns a Sampler object based on the specified options.
     * @param {GenerationConfig} generation_config An object containing options for the sampler.
     * @returns {LogitsSampler} A Sampler object.
     */
    static getSampler(generation_config) {
        // - *greedy decoding*: `num_beams=1` and `do_sample=False`
        // - *contrastive search*: `penalty_alpha>0` and `top_k>1`
        // - *multinomial sampling*: `num_beams=1` and `do_sample=True`
        // - *beam-search decoding*: `num_beams>1` and `do_sample=False`
        // - *beam-search multinomial sampling*: `num_beams>1` and `do_sample=True`
        // - *diverse beam-search decoding*: `num_beams>1` and `num_beam_groups>1`
        // - *constrained beam-search decoding*: `constraints!=None` or `force_words_ids!=None`

        // NOTE: beam search is implemented directly into the generation function
        if (generation_config.do_sample) {
            return new MultinomialSampler(generation_config);
        } else if (generation_config.num_beams > 1) {
            return new BeamSearchSampler(generation_config);
        } else {
            if (generation_config.num_return_sequences > 1) {
                throw Error(
                    `num_return_sequences has to be 1 when doing greedy search, but is ${generation_config.num_return_sequences}.`,
                );
            }
            return new GreedySampler(generation_config);
        }
    }
}

/**
 * Class representing a Greedy Sampler.
 */
class GreedySampler extends LogitsSampler {
    /**
     * Sample the maximum probability of a given logits tensor.
     * @param {Tensor} logits
     * @returns {Promise<[bigint, number][]>} An array with a single tuple, containing the index of the maximum value and a meaningless score (since this is a greedy search).
     */
    async sample(logits) {
        // NOTE: no need to do log_softmax here since we only take the maximum
        const argmax = max(logits.data)[1];

        // Note: score is meaningless in this context, since we are performing
        // greedy search (p = 1 => log(p) = 0)
        return [[BigInt(argmax), 0]];
    }
}

/**
 * Class representing a MultinomialSampler.
 */
class MultinomialSampler extends LogitsSampler {
    /**
     * Creates a new MultinomialSampler.
     * @param {GenerationConfig} generation_config The generation config.
     */
    constructor(generation_config) {
        super(generation_config);

        /**
         * The warpers that select the candidates when `top_p` is set: top-k first (if set), then top-p
         * over the tokens that top-k keeps, as in the Python library. Without `top_p`, this is `null`
         * and `top_k` alone is applied in `sample`.
         * @type {(TopKLogitsWarper|TopPLogitsWarper)[]|null}
         */
        this.warpers = null;

        const { top_k, top_p, num_beams } = generation_config;
        if ((top_p ?? 1.0) < 1.0) {
            // Beam sampling needs at least two candidates per beam
            const min_tokens_to_keep = num_beams > 1 ? 2 : 1;
            this.warpers = [new TopPLogitsWarper(top_p, { min_tokens_to_keep })];
            if ((top_k ?? 0) > 0) {
                this.warpers.unshift(new TopKLogitsWarper(top_k, { min_tokens_to_keep }));
            }
        }
    }

    /**
     * Sample from the logits.
     * @param {Tensor} logits
     * @returns {Promise<[bigint, number][]>}
     */
    async sample(logits) {
        let k = logits.dims.at(-1); // defaults to vocab size
        if (this.warpers) {
            // Filter a copy of the scores, and sample from all tokens that are left. The number
            // of candidates is not capped at `top_k` here, since top-k keeps tokens that are tied
            // with the last one.
            const data = /** @type {Float32Array} */ (logits.data).slice();
            let warped = new Tensor('float32', data, [1, data.length]);
            for (const warper of this.warpers) {
                warped = warper([], warped);
            }
            logits = new Tensor('float32', data, [data.length]);
            k = 0;
            for (let j = 0; j < data.length; ++j) {
                if (data[j] > -Infinity) ++k;
            }
        } else if (this.generation_config.top_k > 0) {
            k = Math.min(this.generation_config.top_k, k);
        }

        // Get top k tokens
        const [v, i] = await topk(logits, k);

        // Compute softmax over logits
        const probabilities = softmax(/** @type {Float32Array} */ (v.data));

        return Array.from({ length: this.generation_config.num_beams }, () => {
            const sampledIndex = this.randomSelect(probabilities);
            return [
                i.data[sampledIndex], // token id
                Math.log(probabilities[sampledIndex]), // score
            ];
        });
    }
}

/**
 * Class representing a BeamSearchSampler.
 */
class BeamSearchSampler extends LogitsSampler {
    /**
     * Sample from the logits.
     * @param {Tensor} logits
     * @returns {Promise<[bigint, number][]>}
     */
    async sample(logits) {
        let k = logits.dims.at(-1); // defaults to vocab size
        if (this.generation_config.top_k > 0) {
            k = Math.min(this.generation_config.top_k, k);
        }

        // Get top k tokens
        const [v, i] = await topk(logits, k);

        // Compute softmax over logits
        const probabilities = softmax(/** @type {Float32Array} */ (v.data));

        return Array.from({ length: this.generation_config.num_beams }, (_, x) => {
            return [
                i.data[x], // token id
                Math.log(probabilities[x]), // score
            ];
        });
    }
}
