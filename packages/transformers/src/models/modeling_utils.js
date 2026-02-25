import { Callable } from '../utils/generic.js';
import { constructSessions, sessionRun } from './session.js';
import { AutoConfig, getCacheShapes } from '../configs.js';
import { Tensor, full_like, cat, zeros_like, ones_like, ones, index_select, index_select_async } from '../utils/tensor.js';
import { DataTypeMap } from '../utils/dtypes.js';

// These will be populated by registry.js
export let MODEL_MAPPING_NAMES = null;

/**
 * Register task mappings (called by registry.js after defining full mappings)
 * @param {Object} mappings - Object with mapping names as keys
 */
export function registerTaskMappings(mappings) {
    MODEL_MAPPING_NAMES = mappings;
}
import { GITHUB_ISSUE_URL } from '../utils/constants.js';
import { getModelJSON } from '../utils/hub.js';
import { Seq2SeqLMOutput } from './modeling_outputs.js';
import {
    LogitsProcessorList,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    NoBadWordsLogitsProcessor,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    TemperatureLogitsWarper,
    ClassifierFreeGuidanceLogitsProcessor,
} from '../generation/logits_process.js';
import { GenerationConfig } from '../generation/configuration_utils.js';
import { EosTokenCriteria, MaxLengthCriteria, StoppingCriteriaList } from '../generation/stopping_criteria.js';
import { LogitsSampler } from '../generation/logits_sampler.js';
import { BeamSearchScorer } from '../generation/beam_search.js';
import { pick } from '../utils/core.js';
import { log_softmax } from '../utils/maths.js';
import { ModelOutput } from './modeling_outputs.js';

/**
 * Converts an array or Tensor of integers to an int64 Tensor.
 * @param {any[]|Tensor} items The input integers to be converted.
 * @returns {Tensor} The int64 Tensor with the converted values.
 * @throws {Error} If the input array is empty or the input is a batched Tensor and not all sequences have the same length.
 * @private
 */
function toI64Tensor(items) {
    if (items instanceof Tensor) {
        return items;
    }
    // items is an array
    if (items.length === 0) {
        throw Error('items must be non-empty');
    }

    if (Array.isArray(items[0])) {
        // batched
        if (items.some((x) => x.length !== items[0].length)) {
            throw Error(
                "Unable to create tensor, you should probably activate truncation and/or padding with 'padding=True' and/or 'truncation=True' to have batched tensors with the same length.",
            );
        }

        return new Tensor('int64', BigInt64Array.from(items.flat().map((x) => BigInt(x))), [
            items.length,
            items[0].length,
        ]);
    } else {
        //flat
        return new Tensor('int64', BigInt64Array.from(items.map((x) => BigInt(x))), [1, items.length]);
    }
}

/**
 * Creates a boolean tensor with a single value.
 * @param {boolean} value The value of the tensor.
 * @returns {Tensor} The boolean tensor.
 * @private
 */
export function boolTensor(value) {
    return new Tensor('bool', [value], [1]);
}

export const MODEL_TYPES = {
    EncoderOnly: 0,
    EncoderDecoder: 1,
    Seq2Seq: 2,
    Vision2Seq: 3,
    DecoderOnly: 4,
    DecoderOnlyWithoutHead: 5,
    MaskGeneration: 6,
    ImageTextToText: 7,
    Musicgen: 8,
    MultiModality: 9,
    Phi3V: 10,
    AudioTextToText: 11,
    AutoEncoder: 12,
    ImageAudioTextToText: 13,
    Supertonic: 14,
    Chatterbox: 15,
};

const MODEL_TYPE_CONFIG = {
    [MODEL_TYPES.DecoderOnly]: {
        can_generate: true,
        forward: decoder_forward,
        prepare_inputs: decoder_prepare_inputs_for_generation,
    },
    [MODEL_TYPES.DecoderOnlyWithoutHead]: {
        can_generate: false,
        forward: decoder_forward,
        prepare_inputs: decoder_prepare_inputs_for_generation,
    },
    [MODEL_TYPES.Seq2Seq]: {
        can_generate: true,
        forward: seq2seq_forward,
        prepare_inputs: encoder_decoder_prepare_inputs_for_generation,
    },
    [MODEL_TYPES.Vision2Seq]: {
        can_generate: true,
        forward: seq2seq_forward,
        prepare_inputs: encoder_decoder_prepare_inputs_for_generation,
    },
    [MODEL_TYPES.Musicgen]: {
        can_generate: true,
        forward: seq2seq_forward,
    },
    [MODEL_TYPES.EncoderDecoder]: {
        can_generate: false,
        forward: seq2seq_forward,
    },
    [MODEL_TYPES.ImageTextToText]: {
        can_generate: true,
        forward: image_text_to_text_forward,
        prepare_inputs: multimodal_text_to_text_prepare_inputs_for_generation,
    },
    [MODEL_TYPES.AudioTextToText]: {
        can_generate: true,
        forward: audio_text_to_text_forward,
        prepare_inputs: multimodal_text_to_text_prepare_inputs_for_generation,
    },
    [MODEL_TYPES.Phi3V]: {
        can_generate: true,
        prepare_inputs: multimodal_text_to_text_prepare_inputs_for_generation,
    },
    [MODEL_TYPES.ImageAudioTextToText]: {
        can_generate: true,
        prepare_inputs: multimodal_text_to_text_prepare_inputs_for_generation,
    },
    [MODEL_TYPES.MultiModality]: {
        can_generate: true,
    },
    [MODEL_TYPES.AutoEncoder]: {
        can_generate: false,
        forward: auto_encoder_forward,
    },
    [MODEL_TYPES.Chatterbox]: {
        can_generate: true,
        forward: encoder_forward,
    },
    default: {
        can_generate: false,
        forward: encoder_forward,
    },
};

export const MODEL_TYPE_MAPPING = new Map();
export const MODEL_NAME_TO_CLASS_MAPPING = new Map();
export const MODEL_CLASS_TO_NAME_MAPPING = new Map();

/**
 * A base class for pre-trained models that provides the model configuration and an ONNX session.
 */
export class PreTrainedModel extends Callable {
    main_input_name = 'input_ids';
    forward_params = ['input_ids', 'attention_mask'];

    _return_dict_in_generate_keys = null;
    /**
     * Creates a new instance of the `PreTrainedModel` class.
     * @param {import('../configs.js').PretrainedConfig} config The model configuration.
     * @param {Record<string, any>} sessions The inference sessions for the model.
     * @param {Record<string, Object>} configs Additional configuration files (e.g., generation_config.json).
     */
    constructor(config, sessions, configs) {
        super();

        this.config = config;
        this.sessions = sessions;
        this.configs = configs;

        const modelName = MODEL_CLASS_TO_NAME_MAPPING.get(this.constructor);
        const modelType = MODEL_TYPE_MAPPING.get(modelName);

        // Get configuration for this model type
        const typeConfig = MODEL_TYPE_CONFIG[modelType] ?? MODEL_TYPE_CONFIG.default;

        this.can_generate = typeConfig.can_generate;
        this._forward = typeConfig.forward;
        this._prepare_inputs_for_generation = typeConfig.prepare_inputs;

        if (this.can_generate) {
            this.forward_params.push('past_key_values');
        }

        /** @type {import('../configs.js').TransformersJSConfig} */
        this.custom_config = this.config['transformers.js_config'] ?? {};
    }

    /**
     * Disposes of all the ONNX sessions that were created during inference.
     * @returns {Promise<unknown[]>} An array of promises, one for each ONNX session that is being disposed.
     * @todo Use https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/FinalizationRegistry
     */
    async dispose() {
        const promises = [];
        for (const session of Object.values(this.sessions)) {
            promises.push(session.release?.());
        }
        return await Promise.all(promises);
    }

    /**
     * Instantiate one of the model classes of the library from a pretrained model.
     *
     * The model class to instantiate is selected based on the `model_type` property of the config object
     * (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible)
     *
     * @param {string} pretrained_model_name_or_path The name or path of the pretrained model. Can be either:
     * - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
     *   Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
     *   user or organization name, like `dbmdz/bert-base-german-cased`.
     * - A path to a *directory* containing model weights, e.g., `./my_model_directory/`.
     * @param {import('../utils/hub.js').PretrainedModelOptions} options Additional options for loading the model.
     *
     * @returns {Promise<PreTrainedModel>} A new instance of the `PreTrainedModel` class.
     */
    static async from_pretrained(
        pretrained_model_name_or_path,
        {
            progress_callback = null,
            config = null,
            cache_dir = null,
            local_files_only = false,
            revision = 'main',
            model_file_name = null,
            subfolder = 'onnx',
            device = null,
            dtype = null,
            use_external_data_format = null,
            session_options = {},
        } = {},
    ) {
        const options = {
            progress_callback,
            config,
            cache_dir,
            local_files_only,
            revision,
            model_file_name,
            subfolder,
            device,
            dtype,
            use_external_data_format,
            session_options,
        };

        const modelName = MODEL_CLASS_TO_NAME_MAPPING.get(this);
        const modelType = MODEL_TYPE_MAPPING.get(modelName);

        config = options.config = await AutoConfig.from_pretrained(pretrained_model_name_or_path, options);

        let info;
        if (modelType === MODEL_TYPES.DecoderOnly) {
            info = await Promise.all([
                constructSessions(
                    pretrained_model_name_or_path,
                    {
                        model: options.model_file_name ?? 'model',
                    },
                    options,
                    'model',
                ),
                get_optional_configs(
                    pretrained_model_name_or_path,
                    {
                        generation_config: 'generation_config.json',
                    },
                    options,
                ),
            ]);
        } else if (modelType === MODEL_TYPES.Seq2Seq || modelType === MODEL_TYPES.Vision2Seq) {
            info = await Promise.all([
                constructSessions(
                    pretrained_model_name_or_path,
                    {
                        model: 'encoder_model',
                        decoder_model_merged: 'decoder_model_merged',
                    },
                    options,
                    'decoder_model_merged',
                ),
                get_optional_configs(
                    pretrained_model_name_or_path,
                    {
                        generation_config: 'generation_config.json',
                    },
                    options,
                ),
            ]);
        } else if (modelType === MODEL_TYPES.MaskGeneration) {
            info = await Promise.all([
                constructSessions(
                    pretrained_model_name_or_path,
                    {
                        model: 'vision_encoder',
                        prompt_encoder_mask_decoder: 'prompt_encoder_mask_decoder',
                    },
                    options,
                ),
            ]);
        } else if (modelType === MODEL_TYPES.EncoderDecoder) {
            info = await Promise.all([
                constructSessions(
                    pretrained_model_name_or_path,
                    {
                        model: 'encoder_model',
                        decoder_model_merged: 'decoder_model_merged',
                    },
                    options,
                    'decoder_model_merged',
                ),
            ]);
        } else if (modelType === MODEL_TYPES.ImageTextToText) {
            const sessions = {
                embed_tokens: 'embed_tokens',
                vision_encoder: 'vision_encoder',
                decoder_model_merged: 'decoder_model_merged',
            };
            if (config.is_encoder_decoder) {
                sessions['model'] = 'encoder_model';
            }
            info = await Promise.all([
                constructSessions(pretrained_model_name_or_path, sessions, options, 'decoder_model_merged'),
                get_optional_configs(
                    pretrained_model_name_or_path,
                    {
                        generation_config: 'generation_config.json',
                    },
                    options,
                ),
            ]);
        } else if (modelType === MODEL_TYPES.AudioTextToText) {
            const sessions = {
                embed_tokens: 'embed_tokens',
                audio_encoder: 'audio_encoder',
                decoder_model_merged: 'decoder_model_merged',
            };
            info = await Promise.all([
                constructSessions(pretrained_model_name_or_path, sessions, options, 'decoder_model_merged'),
                get_optional_configs(
                    pretrained_model_name_or_path,
                    {
                        generation_config: 'generation_config.json',
                    },
                    options,
                ),
            ]);
        } else if (modelType === MODEL_TYPES.ImageAudioTextToText) {
            const sessions = {
                embed_tokens: 'embed_tokens',
                audio_encoder: 'audio_encoder',
                vision_encoder: 'vision_encoder',
                decoder_model_merged: 'decoder_model_merged',
            };
            info = await Promise.all([
                constructSessions(pretrained_model_name_or_path, sessions, options),
                get_optional_configs(
                    pretrained_model_name_or_path,
                    {
                        generation_config: 'generation_config.json',
                    },
                    options,
                ),
            ]);
        } else if (modelType === MODEL_TYPES.Musicgen) {
            info = await Promise.all([
                constructSessions(
                    pretrained_model_name_or_path,
                    {
                        model: 'text_encoder',
                        decoder_model_merged: 'decoder_model_merged',
                        encodec_decode: 'encodec_decode',
                    },
                    options,
                    'decoder_model_merged',
                ),
                get_optional_configs(
                    pretrained_model_name_or_path,
                    {
                        generation_config: 'generation_config.json',
                    },
                    options,
                ),
            ]);
        } else if (modelType === MODEL_TYPES.MultiModality) {
            info = await Promise.all([
                constructSessions(
                    pretrained_model_name_or_path,
                    {
                        prepare_inputs_embeds: 'prepare_inputs_embeds',
                        model: 'language_model',
                        lm_head: 'lm_head',
                        gen_head: 'gen_head',
                        gen_img_embeds: 'gen_img_embeds',
                        image_decode: 'image_decode',
                    },
                    options,
                    'model',
                ),
                get_optional_configs(
                    pretrained_model_name_or_path,
                    {
                        generation_config: 'generation_config.json',
                    },
                    options,
                ),
            ]);
        } else if (modelType === MODEL_TYPES.Phi3V) {
            info = await Promise.all([
                constructSessions(
                    pretrained_model_name_or_path,
                    {
                        prepare_inputs_embeds: 'prepare_inputs_embeds',
                        model: 'model',
                        vision_encoder: 'vision_encoder',
                    },
                    options,
                    'model',
                ),
                get_optional_configs(
                    pretrained_model_name_or_path,
                    {
                        generation_config: 'generation_config.json',
                    },
                    options,
                ),
            ]);
        } else if (modelType === MODEL_TYPES.Chatterbox) {
            info = await Promise.all([
                constructSessions(
                    pretrained_model_name_or_path,
                    {
                        embed_tokens: 'embed_tokens',
                        speech_encoder: 'speech_encoder',
                        model: 'language_model',
                        conditional_decoder: 'conditional_decoder',
                    },
                    options,
                    'model',
                ),
                get_optional_configs(
                    pretrained_model_name_or_path,
                    {
                        generation_config: 'generation_config.json',
                    },
                    options,
                ),
            ]);
        } else if (modelType === MODEL_TYPES.AutoEncoder) {
            info = await Promise.all([
                constructSessions(
                    pretrained_model_name_or_path,
                    {
                        encoder_model: 'encoder_model',
                        decoder_model: 'decoder_model',
                    },
                    options,
                ),
            ]);
        } else if (modelType === MODEL_TYPES.Supertonic) {
            info = await Promise.all([
                constructSessions(
                    pretrained_model_name_or_path,
                    {
                        text_encoder: 'text_encoder',
                        latent_denoiser: 'latent_denoiser',
                        voice_decoder: 'voice_decoder',
                    },
                    options,
                ),
            ]);
        } else {
            // should be MODEL_TYPES.EncoderOnly or MODEL_TYPES.DecoderOnlyWithoutHead
            if (modelType === undefined) {
                const type = modelName ?? config?.model_type;
                if (type !== 'custom') {
                    console.warn(
                        `Model type for '${type}' not found, assuming encoder-only architecture. Please report this at ${GITHUB_ISSUE_URL}.`,
                    );
                }
            }
            info = await Promise.all([
                constructSessions(
                    pretrained_model_name_or_path,
                    {
                        model: options.model_file_name ?? 'model',
                    },
                    options,
                ),
            ]);
        }

        // @ts-ignore
        return new this(config, ...info);
    }

    /**
     * Runs the model with the provided inputs
     * @param {Object} model_inputs Object containing input tensors
     * @returns {Promise<Object>} Object containing output tensors
     */
    async _call(model_inputs) {
        return await this.forward(model_inputs);
    }

    /**
     * Forward method for a pretrained model. If not overridden by a subclass, the correct forward method
     * will be chosen based on the model type.
     * @param {Object} model_inputs The input data to the model in the format specified in the ONNX model.
     * @returns {Promise<Object>} The output data from the model in the format specified in the ONNX model.
     * @throws {Error} This method must be implemented in subclasses.
     */
    async forward(model_inputs) {
        return await this._forward(this, model_inputs);
    }

    /**
     * Get the model's generation config, if it exists.
     * @returns {GenerationConfig|null} The model's generation config if it exists, otherwise `null`.
     */
    get generation_config() {
        return this.configs?.generation_config ?? null;
    }

    /**
     * @param {GenerationConfig} generation_config
     * @param {number} input_ids_seq_length The starting sequence length for the input ids.
     * @returns {LogitsProcessorList}
     * @private
     */
    _get_logits_processor(
        generation_config,
        input_ids_seq_length,
        // encoder_input_ids, TODO
        // prefix_allowed_tokens_fn, TODO
        logits_processor = null,
    ) {
        const processors = new LogitsProcessorList();

        // if (generation_config.diversity_penalty !== null && generation_config.diversity_penalty > 0.0) {
        //     processors.push(new HammingDiversityLogitsProcessor(
        //         generation_config.diversity_penalty,
        //         generation_config.num_beams,
        //         generation_config.num_beam_groups
        //     ));
        // }

        // if (generation_config.encoder_repetition_penalty !== null && generation_config.encoder_repetition_penalty !== 1.0) {
        //     processors.push(new EncoderRepetitionPenaltyLogitsProcessor(
        //         generation_config.encoder_repetition_penalty,
        //         encoder_input_ids
        //     ));
        // }

        if (generation_config.repetition_penalty !== null && generation_config.repetition_penalty !== 1.0) {
            processors.push(new RepetitionPenaltyLogitsProcessor(generation_config.repetition_penalty));
        }

        if (generation_config.no_repeat_ngram_size !== null && generation_config.no_repeat_ngram_size > 0) {
            processors.push(new NoRepeatNGramLogitsProcessor(generation_config.no_repeat_ngram_size));
        }

        // if (generation_config.encoder_no_repeat_ngram_size !== null && generation_config.encoder_no_repeat_ngram_size > 0) {
        //     if (this.config.is_encoder_decoder) {
        //         processors.push(new EncoderNoRepeatNGramLogitsProcessor(
        //             generation_config.encoder_no_repeat_ngram_size,
        //             encoder_input_ids
        //         ));
        //     } else {
        //         throw new Error("It's impossible to use `encoder_no_repeat_ngram_size` with decoder-only architecture");
        //     }
        // }

        if (generation_config.bad_words_ids !== null) {
            processors.push(
                new NoBadWordsLogitsProcessor(generation_config.bad_words_ids, generation_config.eos_token_id),
            );
        }

        if (
            generation_config.min_length !== null &&
            generation_config.eos_token_id !== null &&
            generation_config.min_length > 0
        ) {
            processors.push(new MinLengthLogitsProcessor(generation_config.min_length, generation_config.eos_token_id));
        }

        if (
            generation_config.min_new_tokens !== null &&
            generation_config.eos_token_id !== null &&
            generation_config.min_new_tokens > 0
        ) {
            processors.push(
                new MinNewTokensLengthLogitsProcessor(
                    input_ids_seq_length,
                    generation_config.min_new_tokens,
                    generation_config.eos_token_id,
                ),
            );
        }

        // if (prefix_allowed_tokens_fn !== null) {
        //     processors.push(new PrefixConstrainedLogitsProcessor(
        //         prefix_allowed_tokens_fn,
        //         generation_config.num_beams / generation_config.num_beam_groups
        //     ));
        // }

        if (generation_config.forced_bos_token_id !== null) {
            processors.push(new ForcedBOSTokenLogitsProcessor(generation_config.forced_bos_token_id));
        }

        if (generation_config.forced_eos_token_id !== null) {
            processors.push(
                new ForcedEOSTokenLogitsProcessor(generation_config.max_length, generation_config.forced_eos_token_id),
            );
        }

        // if (generation_config.remove_invalid_values === true) {
        //     processors.push(new InfNanRemoveLogitsProcessor());
        // }

        // if (generation_config.exponential_decay_length_penalty !== null) {
        //     processors.push(new ExponentialDecayLengthPenalty(
        //         generation_config.exponential_decay_length_penalty,
        //         generation_config.eos_token_id,
        //         input_ids_seq_length
        //     ));
        // }

        // if (generation_config.suppress_tokens !== null) {
        //     processors.push(new SuppressTokensLogitsProcessor(generation_config.suppress_tokens));
        // }

        if (generation_config.begin_suppress_tokens !== null) {
            const begin_index =
                input_ids_seq_length > 1 || generation_config.forced_bos_token_id === null
                    ? input_ids_seq_length
                    : input_ids_seq_length + 1;

            processors.push(
                new SuppressTokensAtBeginLogitsProcessor(generation_config.begin_suppress_tokens, begin_index),
            );
        }

        // DEPRECATED: https://github.com/huggingface/transformers/pull/29485
        // if (generation_config.forced_decoder_ids !== null) {
        //     processors.push(new ForceTokensLogitsProcessor(generation_config.forced_decoder_ids));
        // }

        // 8. prepare batched CFG externally
        if (generation_config.guidance_scale !== null && generation_config.guidance_scale > 1) {
            processors.push(new ClassifierFreeGuidanceLogitsProcessor(generation_config.guidance_scale));
        }

        if (generation_config.temperature === 0 && generation_config.do_sample) {
            console.warn(
                '`do_sample` changed to false because `temperature: 0` implies greedy sampling (always selecting the most likely token), which is incompatible with `do_sample: true`.',
            );
            generation_config.do_sample = false;
        }

        if (generation_config.do_sample) {
            if (generation_config.temperature !== null && generation_config.temperature !== 1.0) {
                processors.push(new TemperatureLogitsWarper(generation_config.temperature));
            }
            // TODO: Add TopPLogitsWarper and TopKLogitsWarper
            // if (generation_config.top_k !== null && generation_config.top_k !== 0) {
            //     processors.push(new TopKLogitsWarper(generation_config.top_k));
            // }
            // if (generation_config.top_p !== null && generation_config.top_p < 1.0) {
            //     processors.push(new TopPLogitsWarper(generation_config.top_p));
            // }
        }

        if (logits_processor !== null) {
            processors.extend(logits_processor);
        }

        // `LogitNormalization` should always be the last logit processor, when present
        // if (generation_config.renormalize_logits === true) {
        //     processors.push(new LogitNormalization());
        // }

        return processors;
    }

    /**
     * This function merges multiple generation configs together to form a final generation config to be used by the model for text generation.
     * It first creates an empty `GenerationConfig` object, then it applies the model's own `generation_config` property to it. Finally, if a `generation_config` object was passed in the arguments, it overwrites the corresponding properties in the final config with those of the passed config object.
     * @param {GenerationConfig|null} generation_config A `GenerationConfig` object containing generation parameters.
     * @param {Object} kwargs Additional generation parameters to be used in place of those in the `generation_config` object.
     * @returns {GenerationConfig} The final generation config object to be used by the model for text generation.
     */
    _prepare_generation_config(generation_config, kwargs, cls = GenerationConfig) {
        // Create empty generation config (contains defaults)
        // We pass `this.config` so that if `eos_token_id` or `bos_token_id` exist in the model's config, we will use them
        const config = { ...this.config };
        for (const key of ['decoder', 'generator', 'text_config']) {
            // Special case: some models have generation attributes set in the decoder.
            // Use them if still unset in the generation config.
            if (key in config) {
                Object.assign(config, config[key]);
            }
        }

        const gen_config = new cls(config);

        // Apply model's generation config, if it exists
        Object.assign(gen_config, this.generation_config ?? {});

        // Next, use any generation config specified by the user
        // when calling `generate`
        if (generation_config) {
            Object.assign(gen_config, generation_config);
        }

        // Finally, if any kwargs were passed, use them to overwrite
        if (kwargs) {
            Object.assign(gen_config, pick(kwargs, Object.getOwnPropertyNames(gen_config)));
        }

        return gen_config;
    }

    /**
     *
     * @param {GenerationConfig} generation_config
     * @param {import('../generation/stopping_criteria.js').StoppingCriteria|import('../generation/stopping_criteria.js').StoppingCriteria[]|StoppingCriteriaList} [stopping_criteria=null]
     */
    _get_stopping_criteria(generation_config, stopping_criteria = null) {
        const criteria = new StoppingCriteriaList();

        if (generation_config.max_length !== null) {
            criteria.push(
                new MaxLengthCriteria(generation_config.max_length, this.config.max_position_embeddings ?? null),
            );
        }
        // if (generation_config.max_time !== null) {
        //     criteria.push(new MaxTimeCriteria(generation_config.max_time));
        // }
        if (generation_config.eos_token_id !== null) {
            criteria.push(new EosTokenCriteria(generation_config.eos_token_id));
        }

        if (stopping_criteria) {
            criteria.extend(stopping_criteria);
        }
        return criteria;
    }

    /**
     * Confirms that the model class is compatible with generation.
     * If not, raises an exception that points to the right class to use.
     */
    _validate_model_class() {
        if (!this.can_generate) {
            const generate_compatible_mappings = [
                MODEL_MAPPING_NAMES.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
                // MODEL_MAPPING_NAMES.MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING, // TODO
                MODEL_MAPPING_NAMES.MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES,
                MODEL_MAPPING_NAMES.MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
                MODEL_MAPPING_NAMES.MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES,
            ].filter(Boolean); // Filter out null mappings (in case registry hasn't loaded yet)

            const modelName = MODEL_CLASS_TO_NAME_MAPPING.get(this.constructor);

            const generate_compatible_classes = new Set();
            const modelType = this.config.model_type;
            for (const model_mapping of generate_compatible_mappings) {
                const supported_models = model_mapping?.get(modelType);
                if (supported_models) {
                    generate_compatible_classes.add(supported_models);
                }
            }

            let errorMessage = `The current model class (${modelName}) is not compatible with \`.generate()\`, as it doesn't have a language model head.`;
            if (generate_compatible_classes.size > 0) {
                errorMessage += ` Please use the following class instead: ${[...generate_compatible_classes].join(', ')}`;
            }
            throw Error(errorMessage);
        }
    }

    prepare_inputs_for_generation(...args) {
        if (!this._prepare_inputs_for_generation) {
            throw new Error('prepare_inputs_for_generation is not implemented for this model.');
        }
        return this._prepare_inputs_for_generation(this, ...args);
    }

    /**
     *
     * @param {Object} inputs
     * @param {bigint[][]} inputs.generated_input_ids
     * @param {Object} inputs.outputs
     * @param {Object} inputs.model_inputs
     * @param {boolean} inputs.is_encoder_decoder
     * @returns {Object} The updated model inputs for the next generation iteration.
     */
    _update_model_kwargs_for_generation({ generated_input_ids, outputs, model_inputs, is_encoder_decoder }) {
        // update past_key_values
        model_inputs['past_key_values'] = this.getPastKeyValues(outputs, model_inputs.past_key_values);

        // update inputs for next run
        model_inputs['input_ids'] = new Tensor('int64', generated_input_ids.flat(), [generated_input_ids.length, 1]);

        if (!is_encoder_decoder) {
            // update attention mask
            model_inputs.attention_mask = cat(
                [model_inputs.attention_mask, ones([model_inputs.attention_mask.dims[0], 1])],
                1,
            );
        } else if ('decoder_attention_mask' in model_inputs) {
            // TODO: update decoder attention mask if the model requires it
        }

        // force recreate position_ids in next iteration
        model_inputs['position_ids'] = null;

        return model_inputs;
    }

    /**
     * This function extracts the model-specific `inputs` for generation.
     * @param {Object} params
     * @param {Tensor} [params.inputs=null]
     * @param {number} [params.bos_token_id=null]
     * @param {Record<string, Tensor|number[]>} [params.model_kwargs]
     * @returns {{inputs_tensor: Tensor, model_inputs: Record<string, Tensor>, model_input_name: string}} The model-specific inputs for generation.
     */
    _prepare_model_inputs({ inputs, bos_token_id, model_kwargs }) {
        const model_inputs = pick(model_kwargs, this.forward_params);
        const input_name = this.main_input_name;
        if (input_name in model_inputs) {
            if (inputs) {
                throw new Error(
                    '`inputs`: {inputs}` were passed alongside {input_name} which is not allowed. ' +
                        'Make sure to either pass {inputs} or {input_name}=...',
                );
            }
        } else {
            model_inputs[input_name] = inputs;
        }

        const inputs_tensor = model_inputs[input_name];

        return { inputs_tensor, model_inputs, model_input_name: input_name };
    }

    async _prepare_encoder_decoder_kwargs_for_generation({
        inputs_tensor,
        model_inputs,
        model_input_name,
        generation_config,
    }) {
        if (
            this.sessions['model'].inputNames.includes('inputs_embeds') &&
            !model_inputs.inputs_embeds &&
            '_prepare_inputs_embeds' in this
        ) {
            // Encoder expects `inputs_embeds` instead of `input_ids`
            const { input_ids, pixel_values, attention_mask, ...kwargs } = model_inputs;
            // @ts-ignore
            const prepared_inputs = await this._prepare_inputs_embeds(model_inputs);
            model_inputs = {
                ...kwargs,
                ...pick(prepared_inputs, ['inputs_embeds', 'attention_mask']),
            };
        }
        let { last_hidden_state } = await encoder_forward(this, model_inputs);

        // for classifier free guidance we need to add a 'null' input to our encoder hidden states
        if (generation_config.guidance_scale !== null && generation_config.guidance_scale > 1) {
            last_hidden_state = cat([last_hidden_state, full_like(last_hidden_state, 0.0)], 0);

            if ('attention_mask' in model_inputs) {
                model_inputs['attention_mask'] = cat(
                    [model_inputs['attention_mask'], zeros_like(model_inputs['attention_mask'])],
                    0,
                );
            }
        } else if (model_inputs.decoder_input_ids) {
            // Ensure that the encoder outputs have the same batch size as the decoder inputs,
            // allowing for more efficient batched generation for single inputs
            const decoder_input_ids_batch_size = toI64Tensor(model_inputs.decoder_input_ids).dims[0];
            if (decoder_input_ids_batch_size !== last_hidden_state.dims[0]) {
                if (last_hidden_state.dims[0] !== 1) {
                    throw new Error(
                        `The encoder outputs have a different batch size (${last_hidden_state.dims[0]}) than the decoder inputs (${decoder_input_ids_batch_size}).`,
                    );
                }
                last_hidden_state = cat(
                    Array.from({ length: decoder_input_ids_batch_size }, () => last_hidden_state),
                    0,
                );
            }
        }
        model_inputs['encoder_outputs'] = last_hidden_state;

        return model_inputs;
    }

    /**
     * Prepares `decoder_input_ids` for generation with encoder-decoder models
     * @param {*} param0
     */
    _prepare_decoder_input_ids_for_generation({
        batch_size,
        model_input_name,
        model_kwargs,
        decoder_start_token_id,
        bos_token_id,
        generation_config,
    }) {
        let { decoder_input_ids, ...model_inputs } = model_kwargs;

        // Prepare input ids if the user has not defined `decoder_input_ids` manually.
        if (!(decoder_input_ids instanceof Tensor)) {
            if (!decoder_input_ids) {
                decoder_start_token_id ??= bos_token_id;

                if (this.config.model_type === 'musicgen') {
                    // Custom logic (TODO: move to Musicgen class)
                    decoder_input_ids = Array.from(
                        {
                            // @ts-expect-error TS2339
                            length: batch_size * this.config.decoder.num_codebooks,
                        },
                        () => [decoder_start_token_id],
                    );
                } else if (Array.isArray(decoder_start_token_id)) {
                    if (decoder_start_token_id.length !== batch_size) {
                        throw new Error(
                            `\`decoder_start_token_id\` expcted to have length ${batch_size} but got ${decoder_start_token_id.length}`,
                        );
                    }
                    decoder_input_ids = decoder_start_token_id;
                } else {
                    decoder_input_ids = Array.from(
                        {
                            length: batch_size,
                        },
                        () => [decoder_start_token_id],
                    );
                }
            } else if (!Array.isArray(decoder_input_ids[0])) {
                // Correct batch size
                decoder_input_ids = Array.from(
                    {
                        length: batch_size,
                    },
                    () => decoder_input_ids,
                );
            }
            decoder_input_ids = toI64Tensor(decoder_input_ids);
        }

        model_kwargs['decoder_attention_mask'] = ones_like(decoder_input_ids);

        return { input_ids: decoder_input_ids, model_inputs };
    }

    /**
     * Generates sequences of token ids for models with a language modeling head.
     * @param {import('../generation/parameters.js').GenerationFunctionParameters} options
     * @returns {Promise<ModelOutput|Tensor>} The output of the model, which can contain the generated token ids, attentions, and scores.
     */
    async generate({
        inputs = null,
        generation_config = null,
        logits_processor = null,
        stopping_criteria = null,
        streamer = null,

        // inputs_attention_mask = null,
        ...kwargs
    }) {
        this._validate_model_class();

        // Update generation config with defaults and kwargs
        generation_config = this._prepare_generation_config(generation_config, kwargs);

        // 3. Define model inputs
        let { inputs_tensor, model_inputs, model_input_name } = this._prepare_model_inputs({
            inputs,
            model_kwargs: kwargs,
        });

        const is_encoder_decoder = this.config.is_encoder_decoder;

        // 4. Define other model kwargs
        if (!is_encoder_decoder) {
            // decoder-only models should use left-padding for generation
        } else if (!('encoder_outputs' in model_inputs)) {
            // if model is encoder decoder encoder_outputs are created
            // and added to `model_kwargs`
            model_inputs = await this._prepare_encoder_decoder_kwargs_for_generation({
                inputs_tensor,
                model_inputs,
                model_input_name,
                generation_config,
            });
        }

        // 5. Prepare `input_ids` which will be used for auto-regressive generation
        // TODO: Update to align with HF transformers' implementation
        let input_ids;
        if (is_encoder_decoder) {
            // Generating from the encoder outputs
            ({ input_ids, model_inputs } = this._prepare_decoder_input_ids_for_generation({
                batch_size: model_inputs[model_input_name].dims.at(0),
                model_input_name,
                model_kwargs: model_inputs,
                decoder_start_token_id: generation_config.decoder_start_token_id,
                bos_token_id: generation_config.bos_token_id,
                generation_config,
            }));
        } else {
            input_ids = model_inputs[model_input_name];
        }

        // 6. Prepare `max_length` depending on other stopping criteria.
        let input_ids_length = input_ids.dims.at(-1);

        if (generation_config.max_new_tokens !== null) {
            generation_config.max_length = input_ids_length + generation_config.max_new_tokens;
        }

        // input_ids_length = model_inputs[model_input_name].dims.at(1);
        // // inputs instanceof Tensor ?  : inputs.length;

        // // decoder-only
        // if (input_ids_length === 0) {
        //     throw Error("Must supply a non-empty array of input token ids.")
        // }

        // let decoder_input_ids =
        // generation_config.decoder_input_ids
        // ?? generation_config.decoder_start_token_id
        // ?? generation_config.bos_token_id
        // ?? generation_config.eos_token_id;

        // Update logits processor
        // 8. prepare distribution pre_processing samplers
        const prepared_logits_processor = this._get_logits_processor(
            generation_config,
            input_ids_length,
            logits_processor,
        );

        // 9. prepare stopping criteria
        const prepared_stopping_criteria = this._get_stopping_criteria(generation_config, stopping_criteria);

        // /** @type {number[]} */
        // let eos_token_ids = generation_config.eos_token_id;
        // if (eos_token_ids !== null && !Array.isArray(eos_token_ids)) {
        //     eos_token_ids = [eos_token_ids];
        // }

        const numInputs = model_inputs[model_input_name].dims.at(0);
        const num_beams = generation_config.num_beams;
        const num_beam_groups = generation_config.num_beam_groups;
        const is_beam_search = num_beams > 1;
        const is_group_beam_search = is_beam_search && num_beam_groups > 1;

        let beam_scorer = null;
        let beam_scorers = null;
        /** @type {number[]} */
        let beam_scores;

        if (is_beam_search) {
            // Validate beam search configuration
            if (generation_config.num_return_sequences > num_beams) {
                throw new Error(
                    `num_return_sequences (${generation_config.num_return_sequences}) must be <= num_beams (${num_beams}).`,
                );
            }
            if (num_beam_groups < 1) {
                throw new Error(`num_beam_groups must be >= 1, but got ${num_beam_groups}.`);
            }
            if (num_beam_groups > num_beams) {
                throw new Error(
                    `num_beam_groups (${num_beam_groups}) must be <= num_beams (${num_beams}).`,
                );
            }
            if (num_beams % num_beam_groups !== 0) {
                throw new Error(
                    `num_beams (${num_beams}) must be divisible by num_beam_groups (${num_beam_groups}).`,
                );
            }
            if (is_group_beam_search && generation_config.do_sample) {
                throw new Error('Diverse beam sampling (num_beam_groups > 1 with do_sample = true) is not yet supported.');
            }
            if (generation_config.guidance_scale !== null && generation_config.guidance_scale > 1) {
                throw new Error('Classifier-free guidance (guidance_scale > 1) is not supported with beam search.');
            }
            if (streamer) {
                throw new Error('Streaming is not supported with beam search.');
            }

            if (is_group_beam_search) {
                const group_size = num_beams / num_beam_groups;
                beam_scorers = Array.from({ length: num_beam_groups }, () => new BeamSearchScorer(numInputs, group_size, {
                    length_penalty: generation_config.length_penalty,
                    early_stopping: generation_config.early_stopping,
                    num_return_sequences: group_size,
                    eos_token_id: generation_config.eos_token_id,
                    pad_token_id: generation_config.pad_token_id,
                }));
            } else {
                beam_scorer = new BeamSearchScorer(numInputs, num_beams, {
                    length_penalty: generation_config.length_penalty,
                    early_stopping: generation_config.early_stopping,
                    num_return_sequences: generation_config.num_return_sequences,
                    eos_token_id: generation_config.eos_token_id,
                    pad_token_id: generation_config.pad_token_id,
                });
            }
        }

        const sampler = LogitsSampler.getSampler(generation_config);

        const scores = new Array(is_beam_search ? numInputs * num_beams : numInputs).fill(0);
        /** @type {bigint[][]} */
        const all_input_ids = input_ids.tolist();

        if (is_beam_search) {
            // Expand all_input_ids: [A, B] -> [A, A, A, B, B, B] for num_beams=3
            const expanded_ids = [];
            for (const ids of all_input_ids) {
                for (let b = 0; b < num_beams; ++b) {
                    expanded_ids.push([...ids]);
                }
            }
            all_input_ids.length = 0;
            all_input_ids.push(...expanded_ids);

            // Expand model_inputs tensors by repeating each row num_beams times
            const expanded_indices = [];
            for (let i = 0; i < numInputs; ++i) {
                for (let b = 0; b < num_beams; ++b) {
                    expanded_indices.push(i);
                }
            }
            for (const key of [model_input_name, 'attention_mask', 'decoder_attention_mask', 'encoder_outputs']) {
                const tensor = model_inputs[key];
                if (tensor instanceof Tensor) {
                    if (tensor.location === 'cpu' || tensor.location === 'cpu-pinned') {
                        model_inputs[key] = index_select(tensor, expanded_indices);
                    } else {
                        model_inputs[key] = await index_select_async(tensor, expanded_indices);
                    }
                }
            }

            // Initialize beam_scores: first beam of each input = 0, rest = -1e9
            beam_scores = new Array(numInputs * num_beams).fill(-1e9);
            for (let i = 0; i < numInputs; ++i) {
                const group_size = num_beams / num_beam_groups;
                for (let g = 0; g < num_beam_groups; ++g) {
                    beam_scores[i * num_beams + g * group_size] = 0;
                }
            }
        }

        if (streamer) {
            streamer.put(all_input_ids);
        }

        ////////////////////////////////////////////////////
        // Generic search which handles 4 generation modes:
        // - GenerationMode.GREEDY_SEARCH
        // - GenerationMode.SAMPLE
        // - GenerationMode.BEAM_SEARCH
        // - GenerationMode.BEAM_SAMPLE
        ////////////////////////////////////////////////////
        let outputs;
        let attentions = {};
        let return_dict_items = {};
        while (true) {
            // prepare model inputs
            model_inputs = this.prepare_inputs_for_generation(all_input_ids, model_inputs, generation_config);
            outputs = await this.forward(model_inputs);

            if (generation_config.return_dict_in_generate) {
                if (generation_config.output_attentions) {
                    // Get attentions if they are present
                    const token_attentions = this.getAttentions(outputs);
                    for (const key in token_attentions) {
                        if (!(key in attentions)) {
                            attentions[key] = [];
                        }
                        attentions[key].push(token_attentions[key]);
                    }
                } else if (this._return_dict_in_generate_keys) {
                    Object.assign(return_dict_items, pick(outputs, this._return_dict_in_generate_keys));
                }
            }

            // Logits are of the form [batch_size, out_seq_length, vocab_size]
            // In most cases, this will be [batch_size, 1, vocab_size]
            // So, we select the last token's logits:
            // (equivalent to `logits = outputs.logits[:, -1, :]`)
            // The `.to('float32')` is necessary for models with float16 logits,
            // and is a no-op for float32 logits.
            // TODO: Support float16 sampling in the sampler directly
            const logits = outputs.logits.slice(null, -1, null).to('float32');

            const next_tokens_scores = prepared_logits_processor(all_input_ids, logits);

            /** @type {[bigint][]} */
            const generated_input_ids = [];

            if (is_beam_search) {
                // Beam search: score all candidates across beams, select top 2*num_beams per batch
                const vocab_size = next_tokens_scores.dims.at(-1);
                const total_beams = numInputs * num_beams;

                /** @type {number[]} */
                let next_beam_scores;
                /** @type {bigint[]} */
                let next_beam_tokens;
                /** @type {number[]} */
                let next_beam_indices;

                if (is_group_beam_search) {
                    const group_size = num_beams / num_beam_groups;
                    next_beam_scores = new Array(total_beams).fill(0);
                    next_beam_tokens = new Array(total_beams).fill(0n);
                    next_beam_indices = new Array(total_beams).fill(0);

                    const diversity_penalty = generation_config.diversity_penalty ?? 0;
                    /** @type {Map<bigint, number>[]} */
                    const prev_group_tokens = Array.from({ length: numInputs }, () => new Map());

                    for (let group_idx = 0; group_idx < num_beam_groups; ++group_idx) {
                        const group_offset = group_idx * group_size;

                        const group_all_input_ids = new Array(numInputs * group_size);
                        const group_beam_scores = new Array(numInputs * group_size);
                        for (let batch_idx = 0; batch_idx < numInputs; ++batch_idx) {
                            for (let beam_idx = 0; beam_idx < group_size; ++beam_idx) {
                                const global = batch_idx * num_beams + group_offset + beam_idx;
                                const local = batch_idx * group_size + beam_idx;
                                group_all_input_ids[local] = all_input_ids[global];
                                group_beam_scores[local] = beam_scores[global];
                            }
                        }

                        const group_next_tokens = [];
                        const group_next_indices = [];
                        const group_next_scores = [];

                        for (let batch_idx = 0; batch_idx < numInputs; ++batch_idx) {
                            // Collect candidates across all beams for this batch item
                            /** @type {{score: number, token: bigint, beam_idx: number}[]} */
                            const candidates = [];

                            for (let beam_idx = 0; beam_idx < group_size; ++beam_idx) {
                                const flat_beam = batch_idx * num_beams + group_offset + beam_idx;
                                const logits_data = /** @type {Float32Array} */ (next_tokens_scores[flat_beam].data);
                                const log_probs = log_softmax(logits_data);
                                const beam_score = beam_scores[flat_beam];

                                // Find top 2*group_size candidates per beam for efficiency
                                /** @type {{score: number, token: bigint, beam_idx: number}[]} */
                                const beam_candidates = [];
                                for (let v = 0; v < vocab_size; ++v) {
                                    const token = BigInt(v);
                                    let score = beam_score + log_probs[v];
                                    const penalty_count = prev_group_tokens[batch_idx].get(token) ?? 0;
                                    if (penalty_count > 0) {
                                        score -= diversity_penalty * penalty_count;
                                    }
                                    beam_candidates.push({
                                        score,
                                        token,
                                        beam_idx,
                                    });
                                }
                                beam_candidates.sort((a, b) => b.score - a.score);
                                candidates.push(...beam_candidates.slice(0, 2 * group_size));
                            }

                            // Select top 2*group_size candidates globally for this batch item
                            candidates.sort((a, b) => b.score - a.score);
                            const top_candidates = candidates.slice(0, 2 * group_size);

                            for (const c of top_candidates) {
                                group_next_tokens.push(c.token);
                                group_next_indices.push(c.beam_idx);
                                group_next_scores.push(c.score);
                            }
                        }

                        // Let beam_scorer process: route EOS to hypotheses, select continuing beams
                        const {
                            next_beam_scores: group_next_beam_scores,
                            next_beam_tokens: group_next_beam_tokens,
                            next_beam_indices: group_next_beam_indices,
                        } = beam_scorers[group_idx].process(
                            group_all_input_ids,
                            group_beam_scores,
                            group_next_tokens,
                            group_next_indices,
                            group_next_scores,
                        );

                        // Map group outputs into global beam arrays
                        for (let batch_idx = 0; batch_idx < numInputs; ++batch_idx) {
                            for (let beam_idx = 0; beam_idx < group_size; ++beam_idx) {
                                const global = batch_idx * num_beams + group_offset + beam_idx;
                                const local = batch_idx * group_size + beam_idx;
                                next_beam_scores[global] = group_next_beam_scores[local];
                                next_beam_tokens[global] = group_next_beam_tokens[local];
                                const local_source = group_next_beam_indices[local];
                                const source_in_batch = local_source - batch_idx * group_size;
                                next_beam_indices[global] = batch_idx * num_beams + group_offset + source_in_batch;
                            }
                        }

                        // Track tokens selected by this group for diversity penalty
                        for (let batch_idx = 0; batch_idx < numInputs; ++batch_idx) {
                            const token_counts = prev_group_tokens[batch_idx];
                            for (let beam_idx = 0; beam_idx < group_size; ++beam_idx) {
                                const global = batch_idx * num_beams + group_offset + beam_idx;
                                const token = next_beam_tokens[global];
                                token_counts.set(token, (token_counts.get(token) ?? 0) + 1);
                            }
                        }
                    }
                } else {
                    const all_next_tokens = [];
                    const all_next_indices = [];
                    const all_next_scores = [];
                    const do_beam_sample = generation_config.do_sample;
                    const top_k = generation_config.top_k;

                    for (let batch_idx = 0; batch_idx < numInputs; ++batch_idx) {
                        // Collect candidates across all beams for this batch item
                        /** @type {{score: number, token: bigint, beam_idx: number}[]} */
                        const candidates = [];

                        for (let beam_idx = 0; beam_idx < num_beams; ++beam_idx) {
                            const flat_beam = batch_idx * num_beams + beam_idx;
                            const logits_data = /** @type {Float32Array} */ (next_tokens_scores[flat_beam].data);
                            if (do_beam_sample) {
                                const beam_score = beam_scores[flat_beam];
                                const beam_vocab = logits_data.length;
                                let k = top_k > 0 ? Math.min(top_k, beam_vocab) : beam_vocab;
                                const indices = Array.from({ length: beam_vocab }, (_, i) => i);
                                indices.sort((a, b) => logits_data[b] - logits_data[a]);
                                if (k < indices.length) {
                                    indices.length = k;
                                }
                                const maxLogit = logits_data[indices[0]];
                                const probs = new Array(indices.length);
                                let sum = 0;
                                for (let i = 0; i < indices.length; ++i) {
                                    const val = Math.exp(logits_data[indices[i]] - maxLogit);
                                    probs[i] = val;
                                    sum += val;
                                }
                                for (let i = 0; i < probs.length; ++i) {
                                    probs[i] /= sum;
                                }
                                for (let s = 0; s < num_beams; ++s) {
                                    const sampledIndex = sampler.randomSelect(probs);
                                    const tokenId = indices[sampledIndex];
                                    candidates.push({
                                        score: beam_score + Math.log(probs[sampledIndex]),
                                        token: BigInt(tokenId),
                                        beam_idx,
                                    });
                                }
                            } else {
                                const log_probs = log_softmax(logits_data);

                                // Find top 2*num_beams candidates per beam for efficiency
                                /** @type {{score: number, token: bigint, beam_idx: number}[]} */
                                const beam_candidates = [];
                                for (let v = 0; v < vocab_size; ++v) {
                                    beam_candidates.push({
                                        score: beam_scores[flat_beam] + log_probs[v],
                                        token: BigInt(v),
                                        beam_idx,
                                    });
                                }
                                beam_candidates.sort((a, b) => b.score - a.score);
                                candidates.push(...beam_candidates.slice(0, 2 * num_beams));
                            }
                        }

                        // Select top 2*num_beams candidates globally for this batch item
                        candidates.sort((a, b) => b.score - a.score);
                        const top_candidates = candidates.slice(0, 2 * num_beams);

                        for (const c of top_candidates) {
                            all_next_tokens.push(c.token);
                            all_next_indices.push(c.beam_idx);
                            all_next_scores.push(c.score);
                        }
                    }

                    // Let beam_scorer process: route EOS to hypotheses, select continuing beams
                    ({
                        next_beam_scores,
                        next_beam_tokens,
                        next_beam_indices,
                    } = beam_scorer.process(
                        all_input_ids,
                        beam_scores,
                        all_next_tokens,
                        all_next_indices,
                        all_next_scores,
                    ));
                }

                // Reorder all_input_ids based on beam indices and append new tokens
                const new_all_input_ids = [];
                for (let i = 0; i < total_beams; ++i) {
                    new_all_input_ids.push([...all_input_ids[next_beam_indices[i]], next_beam_tokens[i]]);
                    generated_input_ids.push([next_beam_tokens[i]]);
                }
                all_input_ids.length = 0;
                all_input_ids.push(...new_all_input_ids);

                // Update beam scores
                for (let i = 0; i < total_beams; ++i) {
                    beam_scores[i] = next_beam_scores[i];
                }

                // Reorder KV cache to match beam reordering
                model_inputs['past_key_values'] = await this._reorder_cache(
                    this.getPastKeyValues(outputs, model_inputs.past_key_values),
                    next_beam_indices,
                );
            } else {
                // Greedy / sample: existing behavior
                for (let batch_idx = 0; batch_idx < next_tokens_scores.dims.at(0); ++batch_idx) {
                    const logs = next_tokens_scores[batch_idx];

                    const sampledTokens = await sampler(logs);
                    for (const [newTokenId, logProb] of sampledTokens) {
                        const bigint = BigInt(newTokenId);
                        scores[batch_idx] += logProb;
                        all_input_ids[batch_idx].push(bigint);
                        generated_input_ids.push([bigint]);
                        break;
                    }
                }
            }

            if (streamer) {
                streamer.put(generated_input_ids);
            }

            // Check stopping conditions
            if (is_beam_search) {
                const done = is_group_beam_search
                    ? beam_scorers.every((scorer) => scorer.is_done)
                    : beam_scorer.is_done;
                if (done) {
                    break;
                }
            }
            const stop = prepared_stopping_criteria(all_input_ids);
            if (stop.every((x) => x)) {
                break;
            }

            if (is_beam_search) {
                // For beam search, we already updated past_key_values during reordering.
                // Now update the remaining model inputs.
                model_inputs['input_ids'] = new Tensor('int64', generated_input_ids.flat(), [generated_input_ids.length, 1]);

                if (!is_encoder_decoder) {
                    model_inputs.attention_mask = cat(
                        [model_inputs.attention_mask, ones([model_inputs.attention_mask.dims[0], 1])],
                        1,
                    );
                }

                // Force recreate position_ids in next iteration
                model_inputs['position_ids'] = null;
            } else {
                model_inputs = this._update_model_kwargs_for_generation({
                    generated_input_ids,
                    outputs,
                    model_inputs,
                    is_encoder_decoder,
                });
            }
        }

        if (streamer) {
            streamer.end();
        }

        // Retrieve and dispose all final past key values (including encoder attentions)
        const past_key_values = this.getPastKeyValues(outputs, model_inputs.past_key_values, true);

        let sequences;
        if (is_beam_search) {
            /** @type {bigint[][]} */
            let best_sequences;
            if (is_group_beam_search) {
                const group_size = num_beams / num_beam_groups;
                /** @type {{tokens: bigint[], score: number}[][]} */
                const grouped = Array.from({ length: numInputs }, () => []);

                for (let group_idx = 0; group_idx < num_beam_groups; ++group_idx) {
                    const group_offset = group_idx * group_size;
                    const group_all_input_ids = new Array(numInputs * group_size);
                    const group_beam_scores = new Array(numInputs * group_size);

                    for (let batch_idx = 0; batch_idx < numInputs; ++batch_idx) {
                        for (let beam_idx = 0; beam_idx < group_size; ++beam_idx) {
                            const global = batch_idx * num_beams + group_offset + beam_idx;
                            const local = batch_idx * group_size + beam_idx;
                            group_all_input_ids[local] = all_input_ids[global];
                            group_beam_scores[local] = beam_scores[global];
                        }
                    }

                    const group_results = beam_scorers[group_idx].finalize_with_scores(
                        group_all_input_ids,
                        group_beam_scores,
                    );

                    for (let batch_idx = 0; batch_idx < numInputs; ++batch_idx) {
                        const offset = batch_idx * group_size;
                        for (let i = 0; i < group_size; ++i) {
                            grouped[batch_idx].push(group_results[offset + i]);
                        }
                    }
                }

                best_sequences = [];
                for (let batch_idx = 0; batch_idx < numInputs; ++batch_idx) {
                    const sorted = grouped[batch_idx].sort((a, b) => b.score - a.score);
                    for (let i = 0; i < generation_config.num_return_sequences; ++i) {
                        best_sequences.push(sorted[i].tokens);
                    }
                }
            } else {
                // Finalize beam search: select best hypotheses
                best_sequences = beam_scorer.finalize(all_input_ids, beam_scores);
            }

            // Pad sequences to equal length for tensor creation
            const max_len = Math.max(...best_sequences.map(s => s.length));
            const pad_id = BigInt(generation_config.pad_token_id ?? 0);
            const padded = best_sequences.map(s => {
                const p = [...s];
                while (p.length < max_len) p.push(pad_id);
                return p;
            });
            sequences = new Tensor('int64', padded.flat(), [padded.length, max_len]);
        } else {
            sequences = new Tensor('int64', all_input_ids.flat(), [all_input_ids.length, all_input_ids[0].length]);
        }

        if (generation_config.return_dict_in_generate) {
            return {
                sequences,
                past_key_values,
                ...attentions,
                ...return_dict_items,
            };
        } else {
            // Dispose all remaining tensors
            for (const tensor of Object.values(outputs)) {
                if (tensor.location === 'gpu-buffer') {
                    tensor.dispose();
                }
            }
            return sequences;
        }
    }

    /**
     * Returns an object containing past key values from the given decoder results object.
     *
     * @param {Object} decoderResults The decoder results object.
     * @param {Object} pastKeyValues The previous past key values.
     * @returns {Object} An object containing past key values.
     */
    getPastKeyValues(decoderResults, pastKeyValues, disposeEncoderPKVs = false) {
        const pkvs = Object.create(null);

        for (const name in decoderResults) {
            if (name.startsWith('present')) {
                const newName = name
                    // Hybrid cache architecture
                    .replace('present_ssm', 'past_ssm') // Mamba
                    .replace('present_conv', 'past_conv') // LFM2

                    // Standard cache architecture
                    .replace('present', 'past_key_values');
                const is_encoder_pkv = name.includes('encoder');
                if (is_encoder_pkv && pastKeyValues) {
                    // Optimization introduced by optimum to reuse past key values.
                    // So, we just replace the constant outputs (`decoderResults[name]`) with the previous past key values.
                    // https://github.com/huggingface/optimum/blob/0bf2c05fb7e1182b52d21b703cfc95fd9e4ea3dc/optimum/onnxruntime/base.py#L677-L704
                    pkvs[newName] = pastKeyValues[newName];
                } else {
                    // decoder or using first encoder PKVs
                    pkvs[newName] = decoderResults[name];
                }

                if (pastKeyValues && (!is_encoder_pkv || disposeEncoderPKVs)) {
                    // - Always dispose decoder PKVs
                    // - Only dispose encoder past key values when requested (after generation)
                    const t = pastKeyValues[newName];
                    if (t.location === 'gpu-buffer') {
                        t.dispose();
                    }
                }
            }
        }
        return pkvs;
    }

    /**
     * Reorder the past key values cache to match beam reordering.
     * @param {Object} past_key_values The past key values object.
     * @param {number[]} beam_indices Indices indicating which beam each new position came from.
     * @returns {Promise<Object>} Reordered past key values.
     */
    async _reorder_cache(past_key_values, beam_indices) {
        if (!past_key_values) return past_key_values;
        const reordered = Object.create(null);
        for (const key in past_key_values) {
            const tensor = past_key_values[key];
            const location = tensor.location;
            if (key.includes('encoder')) {
                // Encoder PKVs are shared across beams; pass through
                reordered[key] = tensor;
            } else {
                if (tensor.location === 'cpu' || tensor.location === 'cpu-pinned') {
                    reordered[key] = index_select(tensor, beam_indices);
                } else {
                    reordered[key] = await index_select_async(tensor, beam_indices);
                }
                // Dispose old tensor if it owns GPU resources
                if (location === 'gpu-buffer' || location === 'texture' || location === 'ml-tensor') {
                    tensor.dispose();
                }
            }
        }
        return reordered;
    }

    /**
     * Returns an object containing attentions from the given model output object.
     *
     * @param {Object} model_output The output of the model.
     * @returns {{cross_attentions?: Tensor[]}} An object containing attentions.
     */
    getAttentions(model_output) {
        const attentions = {};

        for (const attnName of ['cross_attentions', 'encoder_attentions', 'decoder_attentions']) {
            for (const name in model_output) {
                if (name.startsWith(attnName)) {
                    if (!(attnName in attentions)) {
                        attentions[attnName] = [];
                    }
                    attentions[attnName].push(model_output[name]);
                }
            }
        }
        return attentions;
    }

    /**
     * Adds past key values to the decoder feeds object. If pastKeyValues is null, creates new tensors for past key values.
     *
     * @param {Object} decoderFeeds The decoder feeds object to add past key values to.
     * @param {Object} pastKeyValues An object containing past key values.
     */
    addPastKeyValues(decoderFeeds, pastKeyValues) {
        if (pastKeyValues) {
            Object.assign(decoderFeeds, pastKeyValues);
        } else {
            const session = this.sessions['decoder_model_merged'] ?? this.sessions['model'];
            const batch_size = (decoderFeeds[this.main_input_name] ?? decoderFeeds.attention_mask)?.dims?.[0] ?? 1;

            const dtype = session?.config?.kv_cache_dtype ?? 'float32';
            const cls = dtype === 'float16' ? DataTypeMap.float16 : DataTypeMap.float32;
            const shapes = getCacheShapes(this.config, { batch_size });
            for (const name in shapes) {
                const size = shapes[name].reduce((a, b) => a * b, 1);
                decoderFeeds[name] = new Tensor(dtype, new cls(size), shapes[name]);
            }
        }
    }

    async encode_image({ pixel_values }) {
        // image_inputs === { pixel_values }
        return (await sessionRun(this.sessions['vision_encoder'], { pixel_values })).image_features;
    }

    async encode_text({ input_ids }) {
        // text_inputs === { input_ids, attention_mask }
        return (await sessionRun(this.sessions['embed_tokens'], { input_ids })).inputs_embeds;
    }

    async encode_audio({ audio_values }) {
        // audio_inputs === { audio_values }
        return (await sessionRun(this.sessions['audio_encoder'], { audio_values })).audio_features;
    }
}

/**
 * Perform forward pass on the seq2seq model (both encoder and decoder).
 * @param {Object} self The seq2seq model object.
 * @param {Object} model_inputs The input object for the model containing encoder and decoder inputs.
 * @returns {Promise<Seq2SeqLMOutput>} Promise that resolves with the output of the seq2seq model.
 * @private
 */
export async function seq2seq_forward(self, model_inputs) {
    let { encoder_outputs, input_ids, decoder_input_ids, ...other_decoder_inputs } = model_inputs;
    // Encode if needed
    if (!encoder_outputs) {
        const encoder_inputs = pick(model_inputs, self.sessions['model'].inputNames);
        // Encoder outputs are not given, so we must compute them.
        encoder_outputs = (await encoder_forward(self, encoder_inputs)).last_hidden_state;
    }

    other_decoder_inputs.input_ids = decoder_input_ids;
    other_decoder_inputs.encoder_hidden_states = encoder_outputs;

    if (self.sessions['decoder_model_merged'].inputNames.includes('encoder_attention_mask')) {
        other_decoder_inputs.encoder_attention_mask = model_inputs.attention_mask;
    }

    return await decoder_forward(self, other_decoder_inputs, true);
}

/**
 * Forward pass of an encoder model.
 * @param {Object} self The encoder model.
 * @param {Object} model_inputs The input data to be used for the forward pass.
 * @returns {Promise<Object>} The model's outputs.
 * @private
 */
export async function encoder_forward(self, model_inputs) {
    const session = self.sessions['model'];
    const encoderFeeds = pick(model_inputs, session.inputNames);

    if (session.inputNames.includes('inputs_embeds') && !encoderFeeds.inputs_embeds) {
        if (!model_inputs.input_ids) {
            throw new Error('Both `input_ids` and `inputs_embeds` are missing in the model inputs.');
        }
        encoderFeeds.inputs_embeds = await self.encode_text({ input_ids: model_inputs.input_ids });
    }
    if (session.inputNames.includes('token_type_ids') && !encoderFeeds.token_type_ids) {
        if (!encoderFeeds.input_ids) {
            throw new Error('Both `input_ids` and `token_type_ids` are missing in the model inputs.');
        }
        // Assign default `token_type_ids` (all zeroes) to the `encoderFeeds` if the model expects it,
        // but they weren't created by the tokenizer.
        encoderFeeds.token_type_ids = zeros_like(encoderFeeds.input_ids);
    }
    if (session.inputNames.includes('pixel_mask') && !encoderFeeds.pixel_mask) {
        if (!encoderFeeds.pixel_values) {
            throw new Error('Both `pixel_values` and `pixel_mask` are missing in the model inputs.');
        }
        // Assign default `pixel_mask` (all ones) to the `encoderFeeds` if the model expects it,
        // but they weren't created by the processor.
        const dims = encoderFeeds.pixel_values.dims;
        encoderFeeds.pixel_mask = ones([dims[0], dims[2], dims[3]]);
    }

    return await sessionRun(session, encoderFeeds);
}

export async function auto_encoder_forward(self, model_inputs) {
    const encoded = await self.encode(model_inputs);
    const decoded = await self.decode(encoded);
    return decoded;
}

/**
 * Forward pass of a decoder model.
 * @param {Object} self The decoder model.
 * @param {Object} model_inputs The input data to be used for the forward pass.
 * @returns {Promise<Object>} The logits and past key values.
 * @private
 */
export async function decoder_forward(self, model_inputs, is_encoder_decoder = false) {
    const session = self.sessions[is_encoder_decoder ? 'decoder_model_merged' : 'model'];

    const { past_key_values, ...new_model_inputs } = model_inputs;

    if (session.inputNames.includes('use_cache_branch')) {
        new_model_inputs.use_cache_branch = boolTensor(!!past_key_values);
    }
    if (
        session.inputNames.includes('position_ids') &&
        new_model_inputs.attention_mask &&
        !new_model_inputs.position_ids
    ) {
        // NOTE: Handle a special case for paligemma/gemma3 models, where positions are 1-indexed
        const start_index = ['paligemma', 'gemma3_text', 'gemma3'].includes(self.config.model_type) ? 1 : 0;
        new_model_inputs.position_ids = create_position_ids(new_model_inputs, past_key_values, start_index);
    }

    // Unpack the `past_key_values` object into model inputs
    self.addPastKeyValues(new_model_inputs, past_key_values);

    // Select only the inputs that are needed for the current session
    const fixed = pick(new_model_inputs, session.inputNames);
    return await sessionRun(session, fixed);
}

/**
 * Abstract forward pass function for image-text-to-text or audio-text-to-text models.
 * @param {Object} self The model object.
 * @param {Object} params Additional parameters.
 * @param {Function} [params.encode_function] The function to encode the modality values.
 * @param {Function} [params.merge_function] The function to merge the modality features with the input embeddings.
 * @param {string} [params.modality_input_name] The modality input name.
 * @param {string} [params.modality_output_name] The modality output name.
 * @param {Tensor} [params.input_ids=null]
 * @param {Tensor} [params.attention_mask=null]
 * @param {Tensor} [params.position_ids=null]
 * @param {Tensor} [params.inputs_embeds=null]
 * @param {Tensor} [params.past_key_values=null]
 * @param {Object} [params.generation_config=null]
 * @param {Object} [params.logits_processor=null]
 * @returns {Promise<Tensor>} The model's output tensor
 * @private
 */
export async function generic_text_to_text_forward(
    self,
    {
        // Generic parameters:
        encode_function,
        merge_function,
        modality_input_name,
        modality_output_name,

        // Produced by the tokenizer/processor:
        input_ids = null,
        attention_mask = null,

        // Used during generation:
        position_ids = null,
        inputs_embeds = null,
        past_key_values = null,

        // Generic generation parameters
        generation_config = null,
        logits_processor = null,

        // Additional parameters
        ...kwargs
    },
) {
    const modality_values = kwargs[modality_input_name];
    if (!inputs_embeds) {
        // 1. Extract the text embeddings.
        inputs_embeds = await self.encode_text({ input_ids, ...kwargs });

        // 2. Possibly, merge text and modality values
        if (modality_values && input_ids.dims[1] !== 1) {
            const modality_features = await encode_function({
                // Pass the modality values under its expected key.
                // The caller knows whether this is audio or image.
                [modality_input_name]: modality_values,
                ...kwargs,
            });
            ({ inputs_embeds, attention_mask } = merge_function({
                [modality_output_name]: modality_features,
                inputs_embeds,
                input_ids,
                attention_mask,
            }));
        } else if (past_key_values && modality_values && input_ids.dims[1] === 1) {
            // This branch handles the cache case.
            const target_length = input_ids.dims[1]; // always 1
            const past_length = Object.values(past_key_values)[0].dims.at(-2);

            attention_mask = cat(
                [
                    ones([input_ids.dims[0], past_length]),
                    attention_mask.slice(null, [attention_mask.dims[1] - target_length, attention_mask.dims[1]]),
                ],
                1,
            );
        }
    }

    if (!position_ids) {
        if (self.config.model_type === 'qwen2_vl') {
            // Special case for qwen2_vl models
            // @ts-ignore
            const { image_grid_thw, video_grid_thw } = kwargs;
            [position_ids] = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask);
        }
    }

    // 3. Call the decoder forward using the updated inputs.
    const outputs = await decoder_forward(
        self,
        {
            inputs_embeds,
            past_key_values,
            attention_mask,
            position_ids,
            generation_config,
            logits_processor,
        },
        true,
    );
    return outputs;
}

/**
 * Forward pass of an audio-text-to-text model.
 * @param {Object} self The audio-text-to-text model.
 * @param {Object} params The inputs for the audio-text-to-text forward pass.
 * @returns {Promise<Tensor>} The model's output tensor.
 * @private
 */
export async function audio_text_to_text_forward(self, params) {
    return await generic_text_to_text_forward(self, {
        ...params,
        modality_input_name: 'audio_values',
        modality_output_name: 'audio_features',
        encode_function: self.encode_audio.bind(self),
        merge_function: self._merge_input_ids_with_audio_features.bind(self),
    });
}

/**
 * Forward pass of an image-text-to-text model.
 * @param {Object} self The image-text-to-text model.
 * @param {Object} params The inputs for the image-text-to-text forward pass.
 * @returns {Promise<Tensor>} The model's output tensor.
 * @private
 */
export async function image_text_to_text_forward(self, params) {
    return await generic_text_to_text_forward(self, {
        ...params,
        modality_input_name: 'pixel_values',
        modality_output_name: 'image_features',
        encode_function: self.encode_image.bind(self),
        merge_function: self._merge_input_ids_with_image_features.bind(self),
    });
}

/**
 * Helper function to perform the following:
 * ```python
 * x = attention_mask.long().cumsum(-1) - 1
 * x.masked_fill_(attention_mask == 0, 1)
 * ```
 * @param {Tensor} attention_mask
 * @returns {{data: BigInt64Array, dims: number[]}}
 */
export function cumsum_masked_fill(attention_mask, start_index = 0) {
    const [bz, seq_len] = attention_mask.dims;
    const attn_mask_data = attention_mask.data;

    const data = new BigInt64Array(attn_mask_data.length);
    for (let i = 0; i < bz; ++i) {
        const start = i * seq_len;
        let sum = BigInt(start_index);
        for (let j = 0; j < seq_len; ++j) {
            const index = start + j;
            if (attn_mask_data[index] === 0n) {
                data[index] = BigInt(1);
            } else {
                // === 1n
                data[index] = sum;
                sum += attn_mask_data[index];
            }
        }
    }
    return { data, dims: attention_mask.dims };
}

/**
 * If the model supports providing position_ids, we create position_ids on the fly for batch generation,
 * by computing the cumulative sum of the attention mask along the sequence length dimension.
 *
 * Equivalent to:
 * ```python
 * position_ids = attention_mask.long().cumsum(-1) - 1
 * position_ids.masked_fill_(attention_mask == 0, 1)
 * if past_key_values:
 *     position_ids = position_ids[:, -input_ids.shape[1] :]
 * ```
 */
export function create_position_ids(model_inputs, past_key_values = null, start_index = 0) {
    const { input_ids, inputs_embeds, attention_mask } = model_inputs;

    const { data, dims } = cumsum_masked_fill(attention_mask, start_index);
    let position_ids = new Tensor('int64', data, dims);
    if (past_key_values) {
        const offset = -(input_ids ?? inputs_embeds).dims.at(1);
        position_ids = position_ids.slice(null, [offset, null]);
    }
    return position_ids;
}

export function decoder_prepare_inputs_for_generation(self, input_ids, model_inputs, generation_config) {
    const past_length = model_inputs.past_key_values ? Object.values(model_inputs.past_key_values)[0].dims.at(-2) : 0;

    if (!model_inputs.attention_mask) {
        // If the attention mask is not provided, we attempt to infer based on provided inputs
        let dims;
        for (const key of ['input_ids', 'inputs_embeds', 'position_ids']) {
            if (model_inputs[key]) {
                dims = model_inputs[key].dims;
                break;
            }
        }
        if (!dims) {
            throw new Error('attention_mask is not provided, and unable to infer its shape from model inputs.');
        }
        model_inputs.attention_mask = ones([dims[0], past_length + dims[1]]);
    }

    if (model_inputs.past_key_values) {
        const { input_ids, attention_mask } = model_inputs;

        // Keep only the unprocessed tokens:
        // 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        // some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        // input)
        if (attention_mask && attention_mask.dims[1] > input_ids.dims[1]) {
            // NOTE: not needed since we only pass the generated tokens to the next forward pass
            // const offset = -(attention_mask.dims[1] - past_length);
            // model_inputs.input_ids = input_ids.slice(null, [offset, null]);
        }
        // 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens.
        // We can discard input_ids based on the past_length.
        else if (past_length < input_ids.dims[1]) {
            // NOTE: Required for phi models.
            // See https://github.com/huggingface/transformers/issues/30809#issuecomment-2111918479 for more information.
            model_inputs.input_ids = input_ids.slice(null, [past_length, null]);
        }
        // 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
        else {
        }
    }

    return model_inputs;
}

export function encoder_decoder_prepare_inputs_for_generation(self, input_ids, model_inputs, generation_config) {
    if (model_inputs.past_key_values) {
        input_ids = input_ids.map((x) => [x.at(-1)]);
    }

    return {
        ...model_inputs,
        decoder_input_ids: toI64Tensor(input_ids),
    };
}

export function multimodal_text_to_text_prepare_inputs_for_generation(self, ...args) {
    if (self.config.is_encoder_decoder) {
        return encoder_decoder_prepare_inputs_for_generation(self, ...args);
    } else {
        return decoder_prepare_inputs_for_generation(self, ...args);
    }
}

export function default_merge_input_ids_with_features({
    modality_token_id,
    inputs_embeds,
    modality_features,
    input_ids,
    attention_mask,
}) {
    const token_positions = input_ids.tolist().map((ids) =>
        ids.reduce((acc, x, idx) => {
            if (x == modality_token_id) acc.push(idx);
            return acc;
        }, []),
    );
    const n_tokens = token_positions.reduce((acc, x) => acc + x.length, 0);
    const n_features = modality_features.dims[0];
    if (n_tokens !== n_features) {
        throw new Error(`Number of tokens and features do not match: tokens: ${n_tokens}, features ${n_features}`);
    }

    // Equivalent to performing a masked_scatter
    let img = 0;
    for (let i = 0; i < token_positions.length; ++i) {
        const tokens = token_positions[i];
        const embeds = inputs_embeds[i];
        for (let j = 0; j < tokens.length; ++j) {
            embeds[tokens[j]].data.set(modality_features[img++].data);
        }
    }
    return { inputs_embeds, attention_mask };
}

export function default_merge_input_ids_with_image_features({
    image_token_id,
    inputs_embeds,
    image_features,
    input_ids,
    attention_mask,
}) {
    return default_merge_input_ids_with_features({
        modality_token_id: image_token_id,
        inputs_embeds,
        modality_features: image_features,
        input_ids,
        attention_mask,
    });
}

export function default_merge_input_ids_with_audio_features({
    audio_token_id,
    inputs_embeds,
    audio_features,
    input_ids,
    attention_mask,
}) {
    return default_merge_input_ids_with_features({
        modality_token_id: audio_token_id,
        inputs_embeds,
        modality_features: audio_features,
        input_ids,
        attention_mask,
    });
}

/**
 * Helper function to load multiple optional configuration files
 * @param {string} pretrained_model_name_or_path The path to the directory containing the config file.
 * @param {Record<string, string>} names The names of the config files to load.
 * @param {import('../utils/hub.js').PretrainedModelOptions} options Additional options for loading the configs.
 * @returns {Promise<Record<string, any>>} A Promise that resolves to a dictionary of configuration objects.
 * @private
 */
export async function get_optional_configs(pretrained_model_name_or_path, names, options) {
    return Object.fromEntries(
        await Promise.all(
            Object.keys(names).map(async (name) => {
                const config = await getModelJSON(pretrained_model_name_or_path, names[name], false, options);
                return [name, config];
            }),
        ),
    );
}
