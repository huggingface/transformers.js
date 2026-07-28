import { Tensor } from '../utils/tensor.js';
import { pick } from '../utils/core.js';
import { logger } from '../utils/logger.js';
import { GenerationConfig } from './configuration_utils.js';
import {
    LogitsProcessorList,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    SuppressTokensLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    NoBadWordsLogitsProcessor,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    TemperatureLogitsWarper,
    ClassifierFreeGuidanceLogitsProcessor,
} from './logits_process.js';
import { EosTokenCriteria, MaxLengthCriteria, StoppingCriteriaList } from './stopping_criteria.js';
import { LogitsSampler } from './logits_sampler.js';

/** @typedef {'greedy'|'multinomial'|'top-k'|'top-p'|'beam-search'} GenerationMode */

/**
 * Resolve the generation configuration independently of an inference runtime.
 *
 * @param {Object} options
 * @param {Object} options.modelConfig
 * @param {Object|null} [options.modelGenerationConfig]
 * @param {Object|null} [options.generationConfig]
 * @param {Object|null} [options.kwargs]
 * @param {typeof GenerationConfig} [options.configClass]
 */
export function prepareGenerationConfig({
    modelConfig,
    modelGenerationConfig = null,
    generationConfig = null,
    kwargs = null,
    configClass = GenerationConfig,
}) {
    const config = { ...modelConfig };
    for (const key of ['decoder', 'generator', 'text_config']) {
        if (key in config) Object.assign(config, config[key]);
    }

    const result = new configClass(config);
    Object.assign(result, modelGenerationConfig ?? {});
    if (generationConfig) Object.assign(result, generationConfig);
    if (kwargs) Object.assign(result, pick(kwargs, Object.getOwnPropertyNames(result)));
    return result;
}

/**
 * Build the complete ordered logits processor list.
 *
 * @param {GenerationConfig} generationConfig
 * @param {number} inputLength
 * @param {import('./logits_process.js').LogitsProcessorList|import('./logits_process.js').LogitsProcessor[]|null} [userProcessors]
 */
export function createLogitsProcessorList(generationConfig, inputLength, userProcessors = null) {
    const processors = new LogitsProcessorList();

    if (generationConfig.repetition_penalty !== null && generationConfig.repetition_penalty !== 1.0) {
        processors.push(new RepetitionPenaltyLogitsProcessor(generationConfig.repetition_penalty));
    }
    if (generationConfig.no_repeat_ngram_size !== null && generationConfig.no_repeat_ngram_size > 0) {
        processors.push(new NoRepeatNGramLogitsProcessor(generationConfig.no_repeat_ngram_size));
    }
    if (generationConfig.bad_words_ids !== null) {
        processors.push(new NoBadWordsLogitsProcessor(generationConfig.bad_words_ids, generationConfig.eos_token_id));
    }
    if (
        generationConfig.min_length !== null &&
        generationConfig.eos_token_id !== null &&
        generationConfig.min_length > 0
    ) {
        processors.push(new MinLengthLogitsProcessor(generationConfig.min_length, generationConfig.eos_token_id));
    }
    if (
        generationConfig.min_new_tokens !== null &&
        generationConfig.eos_token_id !== null &&
        generationConfig.min_new_tokens > 0
    ) {
        processors.push(
            new MinNewTokensLengthLogitsProcessor(
                inputLength,
                generationConfig.min_new_tokens,
                generationConfig.eos_token_id,
            ),
        );
    }
    if (generationConfig.forced_bos_token_id !== null) {
        processors.push(new ForcedBOSTokenLogitsProcessor(generationConfig.forced_bos_token_id));
    }
    if (generationConfig.forced_eos_token_id !== null) {
        processors.push(
            new ForcedEOSTokenLogitsProcessor(generationConfig.max_length, generationConfig.forced_eos_token_id),
        );
    }
    if (generationConfig.suppress_tokens !== null) {
        processors.push(new SuppressTokensLogitsProcessor(generationConfig.suppress_tokens));
    }
    if (generationConfig.begin_suppress_tokens !== null) {
        const beginIndex =
            inputLength > 1 || generationConfig.forced_bos_token_id === null ? inputLength : inputLength + 1;
        processors.push(new SuppressTokensAtBeginLogitsProcessor(generationConfig.begin_suppress_tokens, beginIndex));
    }
    if (generationConfig.guidance_scale !== null && generationConfig.guidance_scale > 1) {
        processors.push(new ClassifierFreeGuidanceLogitsProcessor(generationConfig.guidance_scale));
    }
    if (generationConfig.temperature === 0 && generationConfig.do_sample) {
        logger.warn(
            '`do_sample` changed to false because `temperature: 0` implies greedy sampling (always selecting the most likely token), which is incompatible with `do_sample: true`.',
        );
        generationConfig.do_sample = false;
    }
    if (generationConfig.do_sample && generationConfig.temperature !== null && generationConfig.temperature !== 1.0) {
        processors.push(new TemperatureLogitsWarper(generationConfig.temperature));
    }
    if (userProcessors instanceof LogitsProcessorList) {
        processors.extend(userProcessors.processors);
    } else if (userProcessors !== null) {
        processors.extend(userProcessors);
    }
    return processors;
}

/**
 * Build the complete stopping criteria list.
 *
 * @param {GenerationConfig} generationConfig
 * @param {Object} modelConfig
 * @param {import('./stopping_criteria.js').StoppingCriteria|import('./stopping_criteria.js').StoppingCriteria[]|StoppingCriteriaList|null} [userCriteria]
 */
export function createStoppingCriteriaList(generationConfig, modelConfig, userCriteria = null) {
    const criteria = new StoppingCriteriaList();
    if (generationConfig.max_length !== null) {
        criteria.push(new MaxLengthCriteria(generationConfig.max_length, modelConfig.max_position_embeddings ?? null));
    }
    if (generationConfig.eos_token_id !== null) {
        criteria.push(new EosTokenCriteria(generationConfig.eos_token_id));
    }
    if (userCriteria) criteria.extend(userCriteria);
    return criteria;
}

/**
 * Stateful, inference-runtime-neutral generation policy.
 */
export class GenerationController {
    version = 1;

    /**
     * @param {Object} options
     * @param {Tensor} options.inputIds
     * @param {GenerationConfig} options.generationConfig
     * @param {LogitsProcessorList} options.logitsProcessor
     * @param {StoppingCriteriaList} options.stoppingCriteria
     * @param {import('./streamers.js').BaseStreamer|null} [options.streamer]
     * @param {(outputs: Object, controller: GenerationController) => void} [options.collectOutputs]
     */
    constructor({
        inputIds,
        generationConfig,
        logitsProcessor,
        stoppingCriteria,
        streamer = null,
        collectOutputs = null,
    }) {
        if (!(inputIds instanceof Tensor) || inputIds.dims.length !== 2) {
            throw new TypeError('GenerationController requires a rank-2 input IDs Tensor.');
        }

        this.generationConfig = generationConfig;
        this.logitsProcessor = logitsProcessor;
        this.stoppingCriteria = stoppingCriteria;
        this.streamer = streamer;
        this.collectOutputs = collectOutputs;
        this.batchSize = inputIds.dims[0];
        this.inputLength = inputIds.dims[1];
        /** @type {bigint[][]} */
        this.sequences = inputIds.tolist();
        this.scores = new Array(this.batchSize).fill(0);
        this.done = new Array(this.batchSize).fill(false);
        this.terminal = false;
        this.finalized = false;
        this.aborted = false;

        if (generationConfig.max_new_tokens !== null) {
            generationConfig.max_length = this.inputLength + generationConfig.max_new_tokens;
        }
        this.terminal =
            generationConfig.max_new_tokens === 0 ||
            (generationConfig.max_length !== null && this.inputLength >= generationConfig.max_length);
        if (this.terminal) this.done.fill(true);
        this.sampler = LogitsSampler.getSampler(generationConfig);
        if (streamer) streamer.put(this.sequences.map((tokens) => [...tokens]));
    }

    get allDone() {
        return this.terminal;
    }

    get maxSequenceLength() {
        return this.generationConfig.max_length;
    }

    /**
     * Process CPU-visible logits and commit the selected token.
     *
     * @param {Tensor|{logits: Tensor, outputs?: Object}} input
     */
    async step(input) {
        this.#assertActive();
        const logitsInput = input instanceof Tensor ? input : input.logits;
        const outputs = input instanceof Tensor ? null : (input.outputs ?? null);
        if (!(logitsInput instanceof Tensor)) throw new TypeError('Generation step logits must be a Tensor.');
        if (outputs && this.collectOutputs) this.collectOutputs(outputs, this);

        let logits;
        if (logitsInput.dims.length === 3) {
            logits = logitsInput.slice(null, -1, null).to('float32');
        } else if (logitsInput.dims.length === 2) {
            logits = logitsInput.to('float32');
        } else {
            throw new Error(`Generation logits must have rank 2 or 3, received rank ${logitsInput.dims.length}.`);
        }
        const processed = this.logitsProcessor(this.sequences, logits);
        if (processed.dims[0] !== this.batchSize) {
            throw new Error(
                `Processed generation logits batch size ${processed.dims[0]} does not match ${this.batchSize}.`,
            );
        }
        const tokenIds = new Uint32Array(this.batchSize);
        const tokenScores = new Float64Array(this.batchSize);
        for (let batchIndex = 0; batchIndex < this.batchSize; ++batchIndex) {
            const sampled = await this.sampler(processed[batchIndex]);
            const [tokenId, score] = sampled[0];
            tokenIds[batchIndex] = Number(tokenId);
            tokenScores[batchIndex] = score;
        }
        return this.commit({ tokenIds, scores: tokenScores });
    }

    /**
     * Commit tokens selected by an approved runtime generation plan.
     *
     * @param {{tokenIds: Uint32Array, processedScores?: Float32Array, scores?: Float64Array}} decision
     */
    commit(decision) {
        this.#assertActive();
        if (!(decision.tokenIds instanceof Uint32Array) || decision.tokenIds.length !== this.batchSize) {
            throw new TypeError(`Generation decisions must contain ${this.batchSize} uint32 token IDs.`);
        }

        const generatedInputIds = [];
        for (let index = 0; index < this.batchSize; ++index) {
            const tokenId = BigInt(decision.tokenIds[index]);
            this.sequences[index].push(tokenId);
            this.scores[index] += decision.scores?.[index] ?? 0;
            generatedInputIds.push([tokenId]);
        }
        if (this.streamer) this.streamer.put(generatedInputIds);

        this.done = this.stoppingCriteria(this.sequences, decision.processedScores);
        this.terminal = this.done.every(Boolean);
        const nextTokenIds = new Tensor('int64', generatedInputIds.flat(), [this.batchSize, 1]);
        return {
            nextTokenIds,
            generatedInputIds,
            done: [...this.done],
            allDone: this.terminal,
        };
    }

    /**
     * Compile the currently safe V1 GPU plan. More operations can be added as runtimes advertise them.
     *
     * @param {Object} capabilities
     * @returns {Object|null}
     */
    compileRuntimePlan(capabilities) {
        if (!capabilities?.declarativePlans?.includes('argmax')) return null;
        if (!capabilities?.planModes?.includes('greedy')) return null;
        if (this.generationConfig.do_sample || this.generationConfig.num_beams > 1) return null;
        if (this.generationConfig.output_scores) return null;
        if (this.logitsProcessor.processors.length !== 0) return null;
        return {
            version: 1,
            processors: [],
            sampler: { op: 'argmax' },
            maxNewTokens: Math.max(0, this.maxSequenceLength - this.inputLength),
            pipelineDepth: capabilities.tokenPipeline?.defaultDepth,
        };
    }

    /**
     * @param {Object} [extra]
     */
    finalize(extra = {}) {
        if (this.aborted) throw new Error('Cannot finalize an aborted generation controller.');
        if (this.finalized) throw new Error('Generation controller has already been finalized.');
        if (!this.terminal) throw new Error('Cannot finalize generation before all sequences are done.');
        this.finalized = true;
        if (this.streamer) this.streamer.end();

        // V1 generation is synchronous across rows, so all sequences have equal length.
        const sequences = new Tensor('int64', this.sequences.flat(), [this.sequences.length, this.sequences[0].length]);
        return this.generationConfig.return_dict_in_generate ? { sequences, ...extra } : sequences;
    }

    abort(_reason = undefined) {
        if (this.finalized || this.aborted) return;
        this.aborted = true;
        this.terminal = true;
        if (this.streamer) this.streamer.end();
    }

    #assertActive() {
        if (this.aborted) throw new Error('Generation controller has been aborted.');
        if (this.finalized) throw new Error('Generation controller has already been finalized.');
        if (this.terminal) throw new Error('Generation controller is already complete.');
    }
}

/**
 * Create a controller from model and user generation options.
 *
 * @param {Object} model
 * @param {Tensor} inputIds
 * @param {Object} options
 * @param {(outputs: Object, controller: GenerationController) => void} [collectOutputs]
 */
export function createGenerationController(model, inputIds, options, collectOutputs = null) {
    const {
        generation_config = null,
        logits_processor = null,
        stopping_criteria = null,
        streamer = null,
        ...kwargs
    } = options;
    const generationConfig = prepareGenerationConfig({
        modelConfig: model.config ?? {},
        modelGenerationConfig: model.generation_config ?? null,
        generationConfig: generation_config,
        kwargs,
    });
    if (generationConfig.max_new_tokens !== null) {
        generationConfig.max_length = inputIds.dims.at(-1) + generationConfig.max_new_tokens;
    }
    return new GenerationController({
        inputIds,
        generationConfig,
        logitsProcessor: createLogitsProcessorList(generationConfig, inputIds.dims.at(-1), logits_processor),
        stoppingCriteria: createStoppingCriteriaList(generationConfig, model.config ?? {}, stopping_criteria),
        streamer,
        collectOutputs,
    });
}
