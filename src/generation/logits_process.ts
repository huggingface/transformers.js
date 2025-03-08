
/**
 * @module generation/logits_process
 */

import { Callable } from "../utils/generic.js";
import { Tensor } from "../utils/tensor.js";

import { max, log_softmax } from "../utils/maths.js";

/**
 * Abstract base class for all logit processors that can be applied during generation.
 */
export class LogitsProcessor extends Callable {
    /**
     * Apply the processor to the input logits.
     *
     * @abstract
     * @param {bigint[][]} input_ids The input ids.
     * @param {Tensor} logits The logits to process.
     * @throws {Error} Throws an error if `_call` is not implemented in the subclass.
     */
    _call(input_ids: bigint[][], logits: Tensor) {
        throw Error("`_call` should be implemented in a subclass")
    }
}


/**
 * Abstract base class for all logit warpers that can be applied during generation with multinomial sampling.
 */
export class LogitsWarper extends Callable {
    /**
     * Apply the processor to the input logits.
     *
     * @abstract
     * @param {bigint[][]} input_ids The input ids.
     * @param {Tensor} logits The logits to process.
     * @throws {Error} Throws an error if `_call` is not implemented in the subclass.
     */
    _call(input_ids: bigint[][], logits: Tensor) {
        throw Error("`_call` should be implemented in a subclass")
    }
}


/**
 * A class representing a list of logits processors. A logits processor is a function that modifies the logits
 * output of a language model. This class provides methods for adding new processors and applying all processors to a
 * batch of logits.
 */
export class LogitsProcessorList extends Callable {
    /**
     * The list of logits processors.
     * @type {LogitsProcessor[]}
     */
    processors: LogitsProcessor[];

    /**
     * Constructs a new instance of `LogitsProcessorList`.
     */
    constructor() {
        super();
        this.processors = [];
    }

    /**
     * Adds a new logits processor to the list.
     *
     * @param {LogitsProcessor} item The logits processor function to add.
     */
    push(item: LogitsProcessor) {
        this.processors.push(item);
    }

    /**
     * Adds multiple logits processors to the list.
     *
     * @param {LogitsProcessor[]} items The logits processor functions to add.
     */
    extend(items: LogitsProcessor[]) {
        this.processors.push(...items);
    }

    /**
     * Applies all logits processors in the list to a batch of logits, modifying them in-place.
     *
     * @param {bigint[][]} input_ids The input IDs for the language model.
     * @param {Tensor} logits
     */
    _call(input_ids: bigint[][], logits: Tensor) {
        let toReturn = logits;
        // NOTE: Most processors modify logits inplace
        for (const processor of this.processors) {
            toReturn = processor(input_ids, toReturn);
        }
        return toReturn;
    }

    [Symbol.iterator]() {
        return this.processors.values();
    }
}

// DEPRECATED: https://github.com/huggingface/transformers/pull/29485
// /**
//  * A logits processor that forces a specific token to be generated by the decoder.
//  */
// export class ForceTokensLogitsProcessor extends LogitsProcessor {
//     /**
//      * Constructs a new instance of `ForceTokensLogitsProcessor`.
//      * 
//      * @param {[number, number][]} forced_decoder_ids The ids of tokens that should be forced.
//      */
//     constructor(forced_decoder_ids) {
//         super();
//         // TODO: convert to `new Map(forced_decoder_ids)`
//         this.force_token_map = Object.fromEntries(forced_decoder_ids ?? []);
//     }

//     /**
//      * Apply the processor to the input logits.
//      *
//      * @param {bigint[][]} input_ids The input ids.
//      * @param {Tensor} logits The logits to process.
//      * @returns {Tensor} The processed logits.
//      */
//     _call(input_ids, logits) {
//         console.log('this.force_token_map', this.force_token_map)
//         console.log('call ForceTokensLogitsProcessor', input_ids, logits)
//         console.log('input_ids.length', input_ids.length)
//         let map = this.force_token_map[input_ids.length];
//         if (map) { // There exists a mapping
//             logits.data.fill(-Infinity)
//             logits.data[map] = 0;
//         }
//         console.log('map', map)
//         // throw Error("Not implemented")
//         return logits;
//     }
// }

/**
 * A LogitsProcessor that forces a BOS token at the beginning of the generated sequence.
 */
export class ForcedBOSTokenLogitsProcessor extends LogitsProcessor {
    /**
     * The ID of the beginning-of-sequence token to be forced.
     * @type {number}
     */
    bos_token_id: number;

    /**
     * Create a ForcedBOSTokenLogitsProcessor.
     * @param {number} bos_token_id The ID of the beginning-of-sequence token to be forced.
     */
    constructor(bos_token_id: number) {
        super();
        this.bos_token_id = bos_token_id;
    }

    /**
     * Apply the BOS token forcing to the logits.
     * @param {bigint[][]} input_ids The input IDs.
     * @param {Tensor} logits The logits.
     * @returns {Tensor} The logits with BOS token forcing.
     */
    _call(input_ids: bigint[][], logits: Tensor): Tensor {
        for (let i = 0; i < input_ids.length; ++i) {
            if (input_ids[i].length === 1) {
                const batch_logits_data = /** @type {Float32Array} */(logits[i].data);
                batch_logits_data.fill(-Infinity);
                batch_logits_data[this.bos_token_id] = 0;
            }
        }
        return logits;
    }
}

/**
 * A logits processor that enforces the specified token as the last generated token when `max_length` is reached.
 */
export class ForcedEOSTokenLogitsProcessor extends LogitsProcessor {
    /**
     * The maximum length of the sequence to be generated.
     * @type {number}
     */
    max_length: number;

    /**
     * The id(s) of the *end-of-sequence* token.
     * @type {number[]}
     */
    eos_token_id: number[];

    /**
     * Create a ForcedEOSTokenLogitsProcessor.
     * @param {number} max_length The maximum length of the sequence to be generated.
     * @param {number|number[]} eos_token_id The id(s) of the *end-of-sequence* token.
     */
    constructor(max_length: number, eos_token_id: number | number[]) {
        super();
        this.max_length = max_length;
        this.eos_token_id = Array.isArray(eos_token_id) ? eos_token_id : [eos_token_id];
    }

    /**
     * Apply the processor to input_ids and logits.
     * 
     * @param {bigint[][]} input_ids The input ids.
     * @param {Tensor} logits The logits tensor.
     */
    _call(input_ids: bigint[][], logits: Tensor) {
        for (let i = 0; i < input_ids.length; ++i) {
            if (input_ids[i].length === this.max_length - 1) {
                const batch_logits_data: Float32Array = logits[i].data;
                batch_logits_data.fill(-Infinity);
                for (const eos_token of this.eos_token_id) {
                    batch_logits_data[eos_token] = 0;
                }
            }
        }
        return logits;
    }
}

/**
 * A LogitsProcessor that suppresses a list of tokens as soon as the `generate` function starts
 * generating using `begin_index` tokens. This should ensure that the tokens defined by
 * `begin_suppress_tokens` at not sampled at the begining of the generation.
 */
export class SuppressTokensAtBeginLogitsProcessor extends LogitsProcessor {
    /**
     * The IDs of the tokens to suppress.
     * @type {number[]}
     */
    begin_suppress_tokens: number[];

    /**
     * The number of tokens to generate before suppressing tokens.
     * @type {number}
     */
    begin_index: number;

    /**
     * Create a SuppressTokensAtBeginLogitsProcessor.
     * @param {number[]} begin_suppress_tokens The IDs of the tokens to suppress.
     * @param {number} begin_index The number of tokens to generate before suppressing tokens.
     */
    constructor(begin_suppress_tokens: number[], begin_index: number) {
        super();
        this.begin_suppress_tokens = begin_suppress_tokens;
        this.begin_index = begin_index;
    }

    /**
     * Apply the BOS token forcing to the logits.
     * @param {bigint[][]} input_ids The input IDs.
     * @param {Tensor} logits The logits.
     * @returns {Tensor} The logits with BOS token forcing.
     */
    _call(input_ids: bigint[][], logits: Tensor): Tensor {
        for (let i = 0; i < input_ids.length; ++i) {
            if (input_ids[i].length === this.begin_index) {
                const batch_logits_data = /** @type {Float32Array} */(logits[i].data);
                for (const token_id of this.begin_suppress_tokens) {
                    batch_logits_data[token_id] = -Infinity;
                }
            }
        }
        return logits;
    }
}

/**
 * A LogitsProcessor that handles adding timestamps to generated text.
 */
export class WhisperTimeStampLogitsProcessor extends LogitsProcessor {
    /**
     * The ID of the *end-of-sequence* token.
     * @type {number}
     */
    eos_token_id: number;

    /**
     * The ID of the token that indicates no timestamps are present.
     * @type {number}
     */
    no_timestamps_token_id: number;

    /**
     * The ID of the first timestamp token.
     * @type {number}
     */
    timestamp_begin: number;

    /**
     * The index of the first timestamp token.
     * @type {number}
     */
    begin_index: number;

    /**
     * The maximum index of the first timestamp token.
     * @type {number}
     */
    max_initial_timestamp_index: number;

    /**
     * The maximum index of the first timestamp token.
     * Constructs a new WhisperTimeStampLogitsProcessor.
     * @param {import('../models/whisper/generation_whisper').WhisperGenerationConfig} generate_config The config object passed to the `generate()` method of a transformer model.
     * @param {number[]} init_tokens The initial tokens of the input sequence.
     */
    constructor(generate_config: import('../models/whisper/generation_whisper').WhisperGenerationConfig, init_tokens: number[]) {
        super();
        this.eos_token_id =
            Array.isArray(generate_config.eos_token_id)
                ? generate_config.eos_token_id[0]
                : generate_config.eos_token_id;

        this.no_timestamps_token_id = generate_config.no_timestamps_token_id;
        this.timestamp_begin = this.no_timestamps_token_id + 1;

        this.begin_index = init_tokens.length;
        if (init_tokens.at(-1) === this.no_timestamps_token_id) {
            this.begin_index -= 1;
        }
        this.max_initial_timestamp_index = generate_config.max_initial_timestamp_index;
    }

    /**
     * Modify the logits to handle timestamp tokens.
     * @param {bigint[][]} input_ids The input sequence of tokens.
     * @param {Tensor} logits The logits output by the model.
     * @returns {Tensor} The modified logits.
     */
    _call(input_ids: bigint[][], logits: Tensor): Tensor {
        for (let i = 0; i < input_ids.length; ++i) {
            const batch_logits_data = /** @type {Float32Array} */(logits[i].data);

            // suppress <|notimestamps|> which is handled by without_timestamps
            batch_logits_data[this.no_timestamps_token_id] = -Infinity;

            if (input_ids[i].length === this.begin_index - 1) {
                batch_logits_data.fill(-Infinity);
                batch_logits_data[this.timestamp_begin] = 0;
                continue;
            }

            // timestamps have to appear in pairs, except directly before eos_token; mask logits accordingly
            const seq = input_ids[i].slice(this.begin_index);
            const last_was_timestamp = seq.length >= 1 && seq[seq.length - 1] >= this.timestamp_begin;
            const penultimate_was_timestamp = seq.length < 2 || seq[seq.length - 2] >= this.timestamp_begin;

            if (last_was_timestamp) {
                if (penultimate_was_timestamp) { // has to be non-timestamp
                    batch_logits_data.subarray(this.timestamp_begin).fill(-Infinity);
                } else { // cannot be normal text tokens
                    batch_logits_data.subarray(0, this.eos_token_id).fill(-Infinity);
                }
            }

            // apply the `max_initial_timestamp` option
            if (input_ids[i].length === this.begin_index && this.max_initial_timestamp_index !== null) {
                const last_allowed = this.timestamp_begin + this.max_initial_timestamp_index;
                batch_logits_data.subarray(last_allowed + 1).fill(-Infinity);
            }

            // if sum of probability over timestamps is above any other token, sample timestamp
            const logprobs = log_softmax(batch_logits_data);
            const timestamp_logprob = Math.log(logprobs.subarray(this.timestamp_begin).map(Math.exp).reduce((a, b) => a + b));
            const max_text_token_logprob = max(logprobs.subarray(0, this.timestamp_begin))[0];

            if (timestamp_logprob > max_text_token_logprob) {
                batch_logits_data.subarray(0, this.timestamp_begin).fill(-Infinity);
            }
        }

        return logits;
    }
}

/**
 * A logits processor that disallows ngrams of a certain size to be repeated.
 */
export class NoRepeatNGramLogitsProcessor extends LogitsProcessor {
    /**
     * The no-repeat-ngram size. All ngrams of this size can only occur once.
     * @type {number}
     */
    no_repeat_ngram_size: number;

    /**
     * Create a NoRepeatNGramLogitsProcessor.
     * @param {number} no_repeat_ngram_size The no-repeat-ngram size. All ngrams of this size can only occur once.
     */
    constructor(no_repeat_ngram_size: number) {
        super();
        this.no_repeat_ngram_size = no_repeat_ngram_size;
    }

    /**
     * Generate n-grams from a sequence of token ids.
     * @param {bigint[]} prevInputIds List of previous input ids
     * @returns {Map<string, number[]>} Map of generated n-grams
     */
    getNgrams(prevInputIds: bigint[]): Map<string, number[]> {
        const curLen = prevInputIds.length;

        /**@type {number[][]} */
        const ngrams: number[][] = [];
        for (let j = 0; j < curLen + 1 - this.no_repeat_ngram_size; ++j) {
            const ngram = [];
            for (let k = 0; k < this.no_repeat_ngram_size; ++k) {
                ngram.push(prevInputIds[j + k]);
            }
            ngrams.push(ngram.map(Number));
        }

        /** @type {Map<string, number[]>} */
        const generatedNgram: Map<string, number[]> = new Map();
        for (const ngram of ngrams) {
            const prevNgram = ngram.slice(0, ngram.length - 1);
            const prevNgramKey = JSON.stringify(prevNgram);
            const prevNgramValue = generatedNgram.get(prevNgramKey) ?? [];
            prevNgramValue.push(ngram[ngram.length - 1]);
            generatedNgram.set(prevNgramKey, prevNgramValue);
        }
        return generatedNgram;
    }

    /**
     * Generate n-grams from a sequence of token ids.
     * @param {Map<string, number[]>} bannedNgrams Map of banned n-grams
     * @param {bigint[]} prevInputIds List of previous input ids
     * @returns {number[]} Map of generated n-grams
     */
    getGeneratedNgrams(bannedNgrams: Map<string, number[]>, prevInputIds: bigint[]): number[] {
        const ngramIdx = prevInputIds.slice(prevInputIds.length + 1 - this.no_repeat_ngram_size, prevInputIds.length);
        const banned = bannedNgrams.get(JSON.stringify(ngramIdx.map(Number))) ?? [];
        return banned;
    }

    /**
     * Calculate banned n-gram tokens
     * @param {bigint[]} prevInputIds List of previous input ids
     * @returns {number[]} Map of generated n-grams
     */
    calcBannedNgramTokens(prevInputIds: bigint[]): number[] {
        const bannedTokens = [];
        if (prevInputIds.length + 1 < this.no_repeat_ngram_size) {
            // return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
            return bannedTokens;

        } else {
            const generatedNgrams = this.getNgrams(prevInputIds);
            const bannedTokens = this.getGeneratedNgrams(generatedNgrams, prevInputIds);
            return bannedTokens;
        }
    }

    /**
     * Apply the no-repeat-ngram processor to the logits.
     * @param {bigint[][]} input_ids The input IDs.
     * @param {Tensor} logits The logits.
     * @returns {Tensor} The logits with no-repeat-ngram processing.
     */
    _call(input_ids: bigint[][], logits: Tensor): Tensor {
        for (let i = 0; i < input_ids.length; ++i) {
            const batch_logits_data = /** @type {Float32Array} */(logits[i].data);
            const bannedTokens = this.calcBannedNgramTokens(input_ids[i]);
            for (const token of bannedTokens) {
                batch_logits_data[token] = -Infinity;
            }
        }
        return logits;
    }
}

/**
 * A logits processor that prevents the repetition of previous tokens through a penalty.
 * This penalty is applied at most once per token. Note that, for decoder-only models like most LLMs,
 * the considered tokens include the prompt.
 * 
 * In the original [paper](https://arxiv.org/pdf/1909.05858.pdf), the authors suggest the use of a
 * penalty of around 1.2 to achieve a good balance between truthful generation and lack of repetition.
 * To penalize and reduce repetition, use `penalty` values above 1.0, where a higher value penalizes
 * more strongly. To reward and encourage repetition, use `penalty` values between 0.0 and 1.0, where
 * a lower value rewards more strongly.
 */
export class RepetitionPenaltyLogitsProcessor extends LogitsProcessor {
    /**
     * The parameter for repetition penalty.
     * - 1.0 means no penalty. Above 1.0 penalizes previously generated tokens.
     * - Between 0.0 and 1.0 rewards previously generated tokens.
     * @type {number}
     */
    penalty: number;

    /**
     * Create a RepetitionPenaltyLogitsProcessor.
     * @param {number} penalty The parameter for repetition penalty.
     */
    constructor(penalty: number) {
        super();
        this.penalty = penalty;
    }

    /**
     * Apply the repetition penalty to the logits.
     * @param {bigint[][]} input_ids The input IDs.
     * @param {Tensor} logits The logits.
     * @returns {Tensor} The logits with repetition penalty processing.
     */
    _call(input_ids: bigint[][], logits: Tensor): Tensor {
        for (let i = 0; i < input_ids.length; ++i) {
            const batch_logits_data = /** @type {Float32Array} */(logits[i].data);
            for (const input_id of new Set(input_ids[i])) {
                const token = Number(input_id);
                if (batch_logits_data[token] < 0) {
                    batch_logits_data[token] *= this.penalty;
                } else {
                    batch_logits_data[token] /= this.penalty;
                }
            }
        }

        return logits
    }
}

/**
 * A logits processor that enforces a minimum number of tokens.
 */
export class MinLengthLogitsProcessor extends LogitsProcessor {
    /**
     * The minimum length below which the score of `eos_token_id` is set to negative infinity.
     * @type {number}
     */
    min_length: number;

    /**
     * The ID/IDs of the end-of-sequence token.
     * @type {number[]}
     */
    eos_token_id: number[];

    /**
     * Create a MinLengthLogitsProcessor.
     * @param {number} min_length The minimum length below which the score of `eos_token_id` is set to negative infinity.
     * @param {number|number[]} eos_token_id The ID/IDs of the end-of-sequence token.
     */
    constructor(min_length: number, eos_token_id: number | number[]) {
        super();
        this.min_length = min_length;
        this.eos_token_id = Array.isArray(eos_token_id) ? eos_token_id : [eos_token_id];
    }

    /**
     * Apply logit processor.
     * @param {bigint[][]} input_ids The input IDs.
     * @param {Tensor} logits The logits.
     * @returns {Tensor} The processed logits.
     */
    _call(input_ids: bigint[][], logits: Tensor): Tensor {
        for (let i = 0; i < input_ids.length; ++i) {
            if (input_ids[i].length < this.min_length) {
                const batch_logits_data = /** @type {Float32Array} */(logits[i].data);

                for (const eos_token of this.eos_token_id) {
                    batch_logits_data[eos_token] = -Infinity;
                }
            }
        }

        return logits
    }
}

/**
 * A logits processor that enforces a minimum number of new tokens.
 */
export class MinNewTokensLengthLogitsProcessor extends LogitsProcessor {
    /**
     * The input tokens length.
     * @type {number}
     */
    prompt_length_to_skip: number;

    /**
     * The minimum *new* tokens length below which the score of `eos_token_id` is set to negative infinity.
     * @type {number}
     */
    min_new_tokens: number;

    /**
     * The ID/IDs of the end-of-sequence token.
     * @type {number[]}
     */
    eos_token_id: number[];

    /**
     * Create a MinNewTokensLengthLogitsProcessor.
     * @param {number} prompt_length_to_skip The input tokens length.
     * @param {number} min_new_tokens The minimum *new* tokens length below which the score of `eos_token_id` is set to negative infinity.
     * @param {number|number[]} eos_token_id The ID/IDs of the end-of-sequence token.
     */
    constructor(prompt_length_to_skip: number, min_new_tokens: number, eos_token_id: number | number[]) {
        super();
        this.prompt_length_to_skip = prompt_length_to_skip;
        this.min_new_tokens = min_new_tokens;
        this.eos_token_id = Array.isArray(eos_token_id) ? eos_token_id : [eos_token_id];
    }

    /**
     * Apply logit processor.
     * @param {bigint[][]} input_ids The input IDs.
     * @param {Tensor} logits The logits.
     * @returns {Tensor} The processed logits.
     */
    _call(input_ids: bigint[][], logits: Tensor): Tensor {
        for (let i = 0; i < input_ids.length; ++i) {
            const new_tokens_length = input_ids[i].length - this.prompt_length_to_skip;
            if (new_tokens_length < this.min_new_tokens) {
                const batch_logits_data = /** @type {Float32Array} */(logits[i].data);

                for (const eos_token of this.eos_token_id) {
                    batch_logits_data[eos_token] = -Infinity;
                }
            }
        }
        return logits
    }
}

export class NoBadWordsLogitsProcessor extends LogitsProcessor {
    /**
     * The list of bad words.
     * @type {number[][]}
     */
    bad_words_ids: number[][];

    /**
     * The ID/IDs of the end-of-sequence token.
     * @type {number[]}
     */
    eos_token_id: number[];

    /**
     * Create a `NoBadWordsLogitsProcessor`.
     * @param {number[][]} bad_words_ids List of list of token ids that are not allowed to be generated.
     * @param {number|number[]} eos_token_id The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
     */
    constructor(bad_words_ids: number[][], eos_token_id: number | number[]) {
        super();
        this.bad_words_ids = bad_words_ids;
        this.eos_token_id = Array.isArray(eos_token_id) ? eos_token_id : [eos_token_id];
    }

    /**
     * Apply logit processor.
     * @param {bigint[][]} input_ids The input IDs.
     * @param {Tensor} logits The logits.
     * @returns {Tensor} The processed logits.
     */
    _call(input_ids: bigint[][], logits: Tensor): Tensor {
        for (let i = 0; i < input_ids.length; ++i) {
            const batch_logits_data = /** @type {Float32Array} */(logits[i].data);
            const ids = input_ids[i];
            for (const bad_word_ids of this.bad_words_ids) {
                // Whether to modify the logits of the last token in the bad word id sequence
                let mark = true;

                // For each bad word in the list, if the current sequence of input ids ends with this sequence (excluding the last),
                // then we set the logits of the last bad word id to -Infinity.
                for (let j = 1; j <= bad_word_ids.length - 1 && bad_word_ids.length < ids.length; ++j) {

                    // NOTE: We use != instead of !== to compare bigint and number
                    // @ts-ignore
                    if (bad_word_ids.at(-j - 1) != ids.at(-j)) {
                        // We have found a mismatch
                        mark = false;
                        break;
                    }
                }
                if (mark) {
                    batch_logits_data[bad_word_ids.at(-1)] = -Infinity;
                }
            }
        }
        return logits
    }
}

/**
 * [`LogitsProcessor`] for classifier free guidance (CFG). The scores are split over the batch dimension,
 * where the first half correspond to the conditional logits (predicted from the input prompt) and the second half
 * correspond to the unconditional logits (predicted from an empty or 'null' prompt). The processor computes a
 * weighted average across the conditional and unconditional logits, parameterised by the `guidance_scale`.
 * 
 * See [the paper](https://arxiv.org/abs/2306.05284) for more information.
 */
export class ClassifierFreeGuidanceLogitsProcessor extends LogitsProcessor {
    /**
     * The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`.
     * @type {number}
     */
    guidance_scale: number;

    /**
     * Create a `ClassifierFreeGuidanceLogitsProcessor`.
     * @param {number} guidance_scale The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale > 1`.
     * Higher guidance scale encourages the model to generate samples that are more closely linked to the input
     * prompt, usually at the expense of poorer quality.
     */
    constructor(guidance_scale: number) {
        super();
        if (guidance_scale <= 1) {
            throw new Error(
                `Require guidance scale >1 to use the classifier free guidance processor, got guidance scale ${guidance_scale}.`
            )
        }
        this.guidance_scale = guidance_scale;
    }

    /**
     * Apply logit processor.
     * @param {bigint[][]} input_ids The input IDs.
     * @param {Tensor} logits The logits.
     * @returns {Tensor} The processed logits.
     */
    _call(input_ids: bigint[][], logits: Tensor): Tensor {
        if (logits.dims[0] !== 2 * input_ids.length) {
            throw new Error(
                `Logits should have twice the batch size of the input ids, the first half of batches corresponding to ` +
                `the conditional inputs, and the second half of batches corresponding to the unconditional inputs. Got ` +
                `batch size ${logits.dims[0]} for the logits and ${input_ids.length} for the input ids.`
            )
        }

        const unguided_bsz = input_ids.length;
        const cond_logits = logits.slice([0, unguided_bsz], null);
        const uncond_logits = logits.slice([unguided_bsz, logits.dims[0]], null);

        // Merge into uncond_logits (to save memory). This is equivalent to the following:
        // scores = uncond_logits + (cond_logits - uncond_logits) * guidance_scale
        for (let i = 0; i < uncond_logits.data.length; ++i) {
            uncond_logits.data[i] += (cond_logits.data[i] - uncond_logits.data[i]) * this.guidance_scale;
        }

        return uncond_logits;
    }
}

/**
 * [`LogitsWarper`] for temperature (exponential scaling output probability distribution), which effectively means
 * that it can control the randomness of the predicted tokens. Often used together with [`TopPLogitsWarper`] and [`TopKLogitsWarper`].
 */
export class TemperatureLogitsWarper extends LogitsWarper {
    /**
     * The temperature for temperature (exponential scaling output probability distribution).
     * @type {number}
     */
    temperature: number;

    /**
     * Create a `TemperatureLogitsWarper`.
     * @param {number} temperature Strictly positive float value used to modulate the logits distribution.
     * A value smaller than `1` decreases randomness (and vice versa), with `0` being equivalent to shifting
     * all probability mass to the most likely token.
     */
    constructor(temperature: number) {
        super();

        if (typeof temperature !== 'number' || temperature <= 0) {
            let errorMessage =
                `\`temperature\` (=${temperature}) must be a strictly positive float, otherwise your next token scores will be invalid.`;

            if (temperature === 0) {
                errorMessage += " If you're looking for greedy decoding strategies, set `do_sample=false`."
            }
        }
        this.temperature = temperature;
    }

    /**
     * Apply logit warper.
     * @param {bigint[][]} input_ids The input IDs.
     * @param {Tensor} logits The logits.
     * @returns {Tensor} The processed logits.
     */
    _call(input_ids: bigint[][], logits: Tensor): Tensor {
        const batch_logits_data = /** @type {Float32Array} */(logits.data);
        for (let i = 0; i < batch_logits_data.length; ++i) {
            batch_logits_data[i] /= this.temperature;
        }
        return logits;
    }
}

/**
 * [`LogitsWarper`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off.
 * Often used together with [`TemperatureLogitsWarper`] and [`TopKLogitsWarper`].
 */
export class TopPLogitsWarper extends LogitsWarper {
    /**
     * The probability cutoff for top-p sampling.
     * @type {number}
     */
    top_p: number;

    /**
     * The filter value for top-p sampling.
     * @type {number}
     */
    filter_value: number;

    /**
     * The minimum number of tokens that cannot be filtered.
     * @type {number}
     */
    min_tokens_to_keep: number;

    /**
     * Create a `TopPLogitsWarper`.
     * @param {number} top_p If set to < 1, only the smallest set of most probable tokens with
     * probabilities that add up to `top_p` or higher are kept for generation.
     * @param {Object} options Additional options for the top-p sampling.
     * @param {number} [options.filter_value=-Infinity] All filtered values will be set to this float value.
     * @param {number} [options.min_tokens_to_keep=1] Minimum number of tokens that cannot be filtered.
     */
    constructor(top_p: number, {
        filter_value = -Infinity,
        min_tokens_to_keep = 1,
    }: { filter_value?: number; min_tokens_to_keep?: number; } = {}) {
        super();
        if (top_p < 0 || top_p > 1.0) {
            throw new Error(`\`top_p\` must be a float > 0 and < 1, but is ${top_p}`)
        }
        if (!Number.isInteger(min_tokens_to_keep) || min_tokens_to_keep < 1) {
            throw new Error(`\`min_tokens_to_keep\` must be a positive integer, but is ${min_tokens_to_keep}`)
        }

        this.top_p = top_p
        this.filter_value = filter_value
        this.min_tokens_to_keep = min_tokens_to_keep
    }
}

/**
 * [`LogitsWarper`] that performs top-k, i.e. restricting to the k highest probability elements.
 * Often used together with [`TemperatureLogitsWarper`] and [`TopPLogitsWarper`].
 */
export class TopKLogitsWarper extends LogitsWarper {
    /**
     * The number of top tokens to keep.
     * @type {number}
     */
    top_k: number;

    /**
     * The filter value for top-k sampling.
     * @type {number}
     */
    filter_value: number;

    /**
     * Create a `TopKLogitsWarper`.
     * @param {number} top_k If set to > 0, only the top `top_k` tokens are kept for generation.
     * @param {Object} options Additional options for the top-k sampling.
     * @param {number} [options.filter_value=-Infinity] All filtered values will be set to this float value.
     * @param {number} [options.min_tokens_to_keep=1] Minimum number of tokens that cannot be filtered.
     */
    constructor(top_k: number, {
        filter_value = -Infinity,
        min_tokens_to_keep = 1,
    }: { filter_value?: number; min_tokens_to_keep?: number; } = {}) {
        super();
        if (!Number.isInteger(top_k) || top_k < 0) {
            throw new Error(`\`top_k\` must be a positive integer, but is ${top_k}`)
        }

        this.top_k = Math.max(top_k, min_tokens_to_keep)
        this.filter_value = filter_value
    }
}