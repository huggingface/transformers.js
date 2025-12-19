/**
 * @file Definitions of all models available in Transformers.js.
 *
 * **Example:** Load and run an `AutoModel`.
 *
 * ```javascript
 * import { AutoModel, AutoTokenizer } from '@huggingface/transformers';
 *
 * let tokenizer = await AutoTokenizer.from_pretrained('Xenova/bert-base-uncased');
 * let model = await AutoModel.from_pretrained('Xenova/bert-base-uncased');
 *
 * let inputs = await tokenizer('I love transformers!');
 * let { logits } = await model(inputs);
 * // Tensor {
 * //     data: Float32Array(183132) [-7.117443084716797, -7.107812881469727, -7.092104911804199, ...]
 * //     dims: (3) [1, 6, 30522],
 * //     type: "float32",
 * //     size: 183132,
 * // }
 * ```
 *
 * We also provide other `AutoModel`s (listed below), which you can use in the same way as the Python library. For example:
 *
 * **Example:** Load and run an `AutoModelForSeq2SeqLM`.
 * ```javascript
 * import { AutoModelForSeq2SeqLM, AutoTokenizer } from '@huggingface/transformers';
 *
 * let tokenizer = await AutoTokenizer.from_pretrained('Xenova/t5-small');
 * let model = await AutoModelForSeq2SeqLM.from_pretrained('Xenova/t5-small');
 *
 * let { input_ids } = await tokenizer('translate English to German: I love transformers!');
 * let outputs = await model.generate(input_ids);
 * let decoded = tokenizer.decode(outputs[0], { skip_special_tokens: true });
 * // 'Ich liebe Transformatoren!'
 * ```
 *
 * @module models
 */

import { AutoConfig, getCacheShapes } from './configs.js';

import {
    deviceToExecutionProviders,
    createInferenceSession,
    isONNXTensor,
    isONNXProxy,
    runInferenceSession,
} from './backends/onnx.js';
import {
    DATA_TYPES,
    DEFAULT_DEVICE_DTYPE_MAPPING,
    DEFAULT_DTYPE_SUFFIX_MAPPING,
    isWebGpuFp16Supported,
} from './utils/dtypes.js';

import { Callable } from './utils/generic.js';

import { mergeArrays, pick } from './utils/core.js';

import { getModelFile, getModelJSON, MAX_EXTERNAL_DATA_CHUNKS } from './utils/hub.js';

import { GITHUB_ISSUE_URL } from './utils/constants.js';

import {
    LogitsProcessorList,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    WhisperTimeStampLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    NoBadWordsLogitsProcessor,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    TemperatureLogitsWarper,
    ClassifierFreeGuidanceLogitsProcessor,
} from './generation/logits_process.js';

import { GenerationConfig } from './generation/configuration_utils.js';

import {
    cat,
    mean,
    zeros,
    zeros_like,
    ones,
    ones_like,
    full,
    full_like,
    stack,
    std_mean,
    Tensor,
    DataTypeMap,
    randn,
    boolTensor,
    toI64Tensor,
} from './utils/tensor.js';
import { RawImage } from './utils/image.js';

import { dynamic_time_warping, max, medianFilter } from './utils/maths.js';
import { EosTokenCriteria, MaxLengthCriteria, StoppingCriteriaList } from './generation/stopping_criteria.js';
import { LogitsSampler } from './generation/logits_sampler.js';
import { apis, env } from './env.js';
import {
    decoderForward,
    decoder_prepare_inputs_for_generation,
    seq2seqForward,
    encoder_decoder_prepare_inputs_for_generation,
    imageTextToTextForward,
    multimodal_text_to_text_prepare_inputs_for_generation,
    audioTextToTextForward,
    multimodality_prepare_inputs_for_generation,
    autoEncoderForward,
    chatterbox_prepare_inputs_for_generation,
    encoderForward,
} from './models/utils.js';

import { WhisperGenerationConfig } from './models/whisper/generation_whisper.js';
import { whisper_language_to_code } from './models/whisper/common_whisper.js';

//////////////////////////////////////////////////
// Model types: used internally
const MODEL_TYPES = {
    EncoderOnly: 0,
    EncoderDecoder: 1,
    Seq2Seq: 2,
    Vision2Seq: 3,
    DecoderOnly: 4,
    MaskGeneration: 5,
    ImageTextToText: 6,
    Musicgen: 7,
    MultiModality: 8,
    Phi3V: 9,
    AudioTextToText: 10,
    AutoEncoder: 11,
    ImageAudioTextToText: 12,
    Supertonic: 13,
    Chatterbox: 14,
};
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// Helper functions

// NOTE: These will be populated fully later
const MODEL_TYPE_MAPPING = new Map();
const MODEL_NAME_TO_CLASS_MAPPING = new Map();
const MODEL_CLASS_TO_NAME_MAPPING = new Map();

//////////////////////////////////////////////////

//////////////////////////////////////////////////
/**
 * A base class for pre-trained models that provides the model configuration and an ONNX session.
 */
export class PreTrainedModel extends Callable {
    main_input_name = 'input_ids';
    forward_params = ['input_ids', 'attention_mask'];

    _return_dict_in_generate_keys = null;
    /**
     * Creates a new instance of the `PreTrainedModel` class.
     * @param {import('./configs.js').PretrainedConfig} config The model configuration.
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

        this.can_generate = false;
        this._forward = null;

        this._prepare_inputs_for_generation = null;
        switch (modelType) {
            case MODEL_TYPES.DecoderOnly:
                this.can_generate = true;
                this._forward = decoderForward;
                this._prepare_inputs_for_generation = decoder_prepare_inputs_for_generation;
                break;
            case MODEL_TYPES.Seq2Seq:
            case MODEL_TYPES.Vision2Seq:
            case MODEL_TYPES.Musicgen:
                this.can_generate = true;

                this._forward = seq2seqForward;
                this._prepare_inputs_for_generation = encoder_decoder_prepare_inputs_for_generation;
                break;

            case MODEL_TYPES.EncoderDecoder:
                this._forward = seq2seqForward;
                break;
            case MODEL_TYPES.ImageTextToText:
                this.can_generate = true;
                this._forward = imageTextToTextForward;
                this._prepare_inputs_for_generation = multimodal_text_to_text_prepare_inputs_for_generation;
                break;
            case MODEL_TYPES.AudioTextToText:
                this.can_generate = true;
                this._forward = audioTextToTextForward;
                this._prepare_inputs_for_generation = multimodal_text_to_text_prepare_inputs_for_generation;
                break;
            case MODEL_TYPES.Phi3V:
            case MODEL_TYPES.ImageAudioTextToText:
                this.can_generate = true;
                this._prepare_inputs_for_generation = multimodal_text_to_text_prepare_inputs_for_generation;
                break;
            case MODEL_TYPES.MultiModality:
                this.can_generate = true;
                this._prepare_inputs_for_generation = multimodality_prepare_inputs_for_generation;
                break;
            case MODEL_TYPES.AutoEncoder:
                this._forward = autoEncoderForward;
                break;
            case MODEL_TYPES.Chatterbox:
                this.can_generate = true;
                this._prepare_inputs_for_generation = chatterbox_prepare_inputs_for_generation;
            default:
                // should be MODEL_TYPES.EncoderOnly
                this._forward = encoderForward;
                break;
        }

        if (this.can_generate) {
            this.forward_params.push('past_key_values');
        }

        /** @type {import('./configs.js').TransformersJSConfig} */
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
     * @param {import('./utils/hub.js').PretrainedModelOptions} options Additional options for loading the model.
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
        let options = {
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
                getOptionalConfigs(
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
                getOptionalConfigs(
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
                getOptionalConfigs(
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
                getOptionalConfigs(
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
                getOptionalConfigs(
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
                getOptionalConfigs(
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
                getOptionalConfigs(
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
                getOptionalConfigs(
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
                getOptionalConfigs(
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
            // should be MODEL_TYPES.EncoderOnly
            if (modelType !== MODEL_TYPES.EncoderOnly) {
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
     * @param {StoppingCriteriaList} [stopping_criteria=null]
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
                MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
                // MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING, // TODO
                MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES,
                MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
                MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES,
            ];

            const modelName = MODEL_CLASS_TO_NAME_MAPPING.get(this.constructor);

            const generate_compatible_classes = new Set();
            const modelType = this.config.model_type;
            for (const model_mapping of generate_compatible_mappings) {
                const supported_models = model_mapping.get(modelType);
                if (supported_models) {
                    generate_compatible_classes.add(supported_models[0]);
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
        let { last_hidden_state } = await encoderForward(this, model_inputs);

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
     * @param {import('./generation/parameters.js').GenerationFunctionParameters} options
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

        // TODO:
        // done is a list of booleans to keep track of which inputs are done
        // const done = new Array(numInputs).fill(false);
        // For efficiency purposes, we remove completed rows from model_inputs
        // when the beam is complete, and we keep track of the row index
        // const rowIndexToBatchIndex = new Map();

        const sampler = LogitsSampler.getSampler(generation_config);

        // TODO make > numInputs
        const scores = new Array(numInputs).fill(0);
        /** @type {bigint[][]} */
        const all_input_ids = input_ids.tolist();
        if (streamer) {
            streamer.put(all_input_ids);
        }
        // const all_generated_input_ids = Array.from({ length: numInputs }, () => []);

        // NOTE: For now, we don't support spawning new beams
        // TODO: when we do, we simply copy past key values and accumulate into single large tensor

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
            // const new_kv_cache = [];// NOTE: Only used for beam search when concatenating new kv
            // Loop over each batch
            for (let batch_idx = 0; batch_idx < next_tokens_scores.dims.at(0); ++batch_idx) {
                const logs = next_tokens_scores[batch_idx];

                const sampledTokens = await sampler(logs);
                for (const [newTokenId, logProb] of sampledTokens) {
                    const bigint = BigInt(newTokenId);
                    // TODO: If branching, use previous beam as a starting point
                    // update generated ids, model inputs, and length for next step
                    scores[batch_idx] += logProb;
                    all_input_ids[batch_idx].push(bigint);
                    generated_input_ids.push([bigint]);

                    // TODO: Support beam search
                    break;
                }
            }
            if (streamer) {
                streamer.put(generated_input_ids);
            }

            const stop = prepared_stopping_criteria(all_input_ids);
            if (stop.every((x) => x)) {
                break;
            }

            model_inputs = this._update_model_kwargs_for_generation({
                generated_input_ids,
                outputs,
                model_inputs,
                is_encoder_decoder,
            });
        }

        if (streamer) {
            streamer.end();
        }

        // Retrieve and dispose all final past key values (including encoder attentions)
        const past_key_values = this.getPastKeyValues(outputs, model_inputs.past_key_values, true);

        // TODO: ensure all_input_ids is padded correctly...
        const sequences = new Tensor('int64', all_input_ids.flat(), [all_input_ids.length, all_input_ids[0].length]);

        if (generation_config.return_dict_in_generate) {
            return {
                sequences,
                past_key_values,
                ...attentions,
                ...return_dict_items,
                // TODO:
                // scores,
                // logits,
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



//////////////////////////////////////////////////
// Phi models

//////////////////////////////////////////////////

//////////////////////////////////////////////////

//////////////////////////////////////////////////

//////////////////////////////////////////////////


//////////////////////////////////////////////////
export class VitMattePreTrainedModel extends PreTrainedModel {}

/**
 * ViTMatte framework leveraging any vision backbone e.g. for ADE20k, CityScapes.
 *
 * **Example:** Perform image matting with a `VitMatteForImageMatting` model.
 * ```javascript
 * import { AutoProcessor, VitMatteForImageMatting, RawImage } from '@huggingface/transformers';
 *
 * // Load processor and model
 * const processor = await AutoProcessor.from_pretrained('Xenova/vitmatte-small-distinctions-646');
 * const model = await VitMatteForImageMatting.from_pretrained('Xenova/vitmatte-small-distinctions-646');
 *
 * // Load image and trimap
 * const image = await RawImage.fromURL('https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/vitmatte_image.png');
 * const trimap = await RawImage.fromURL('https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/vitmatte_trimap.png');
 *
 * // Prepare image + trimap for the model
 * const inputs = await processor(image, trimap);
 *
 * // Predict alpha matte
 * const { alphas } = await model(inputs);
 * // Tensor {
 * //   dims: [ 1, 1, 640, 960 ],
 * //   type: 'float32',
 * //   size: 614400,
 * //   data: Float32Array(614400) [ 0.9894027709960938, 0.9970508813858032, ... ]
 * // }
 * ```
 *
 * You can visualize the alpha matte as follows:
 * ```javascript
 * import { Tensor, cat } from '@huggingface/transformers';
 *
 * // Visualize predicted alpha matte
 * const imageTensor = image.toTensor();
 *
 * // Convert float (0-1) alpha matte to uint8 (0-255)
 * const alphaChannel = alphas
 *   .squeeze(0)
 *   .mul_(255)
 *   .clamp_(0, 255)
 *   .round_()
 *   .to('uint8');
 *
 * // Concatenate original image with predicted alpha
 * const imageData = cat([imageTensor, alphaChannel], 0);
 *
 * // Save output image
 * const outputImage = RawImage.fromTensor(imageData);
 * outputImage.save('output.png');
 * ```
 */
export class VitMatteForImageMatting extends VitMattePreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new ImageMattingOutput(await super._call(model_inputs));
    }
}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
export class MobileViTPreTrainedModel extends PreTrainedModel {}
export class MobileViTModel extends MobileViTPreTrainedModel {}
export class MobileViTForImageClassification extends MobileViTPreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
// TODO: MobileViTForSemanticSegmentation

//////////////////////////////////////////////////

//////////////////////////////////////////////////
export class MobileViTV2PreTrainedModel extends PreTrainedModel {}
export class MobileViTV2Model extends MobileViTV2PreTrainedModel {}
export class MobileViTV2ForImageClassification extends MobileViTV2PreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
// TODO: MobileViTV2ForSemanticSegmentation

//////////////////////////////////////////////////

//////////////////////////////////////////////////
export class OwlViTPreTrainedModel extends PreTrainedModel {}
export class OwlViTModel extends OwlViTPreTrainedModel {}
export class OwlViTForObjectDetection extends OwlViTPreTrainedModel {}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
export class Owlv2PreTrainedModel extends PreTrainedModel {}
export class Owlv2Model extends Owlv2PreTrainedModel {}
export class Owlv2ForObjectDetection extends Owlv2PreTrainedModel {}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// Beit Models
export class BeitPreTrainedModel extends PreTrainedModel {}
export class BeitModel extends BeitPreTrainedModel {}
export class BeitForImageClassification extends BeitPreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
export class DetrPreTrainedModel extends PreTrainedModel {}
export class DetrModel extends DetrPreTrainedModel {}
export class DetrForObjectDetection extends DetrPreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new DetrObjectDetectionOutput(await super._call(model_inputs));
    }
}

export class DetrForSegmentation extends DetrPreTrainedModel {
    /**
     * Runs the model with the provided inputs
     * @param {Object} model_inputs Model inputs
     * @returns {Promise<DetrSegmentationOutput>} Object containing segmentation outputs
     */
    async _call(model_inputs) {
        return new DetrSegmentationOutput(await super._call(model_inputs));
    }
}

export class DetrObjectDetectionOutput extends ModelOutput {
    /**
     * @param {Object} output The output of the model.
     * @param {Tensor} output.logits Classification logits (including no-object) for all queries.
     * @param {Tensor} output.pred_boxes Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height).
     * These values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding possible padding).
     */
    constructor({ logits, pred_boxes }) {
        super();
        this.logits = logits;
        this.pred_boxes = pred_boxes;
    }
}

export class DetrSegmentationOutput extends ModelOutput {
    /**
     * @param {Object} output The output of the model.
     * @param {Tensor} output.logits The output logits of the model.
     * @param {Tensor} output.pred_boxes Predicted boxes.
     * @param {Tensor} output.pred_masks Predicted masks.
     */
    constructor({ logits, pred_boxes, pred_masks }) {
        super();
        this.logits = logits;
        this.pred_boxes = pred_boxes;
        this.pred_masks = pred_masks;
    }
}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
export class RTDetrPreTrainedModel extends PreTrainedModel {}
export class RTDetrModel extends RTDetrPreTrainedModel {}
export class RTDetrForObjectDetection extends RTDetrPreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new RTDetrObjectDetectionOutput(await super._call(model_inputs));
    }
}

export class RTDetrObjectDetectionOutput extends ModelOutput {
    /**
     * @param {Object} output The output of the model.
     * @param {Tensor} output.logits Classification logits (including no-object) for all queries.
     * @param {Tensor} output.pred_boxes Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height).
     * These values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding possible padding).
     */
    constructor({ logits, pred_boxes }) {
        super();
        this.logits = logits;
        this.pred_boxes = pred_boxes;
    }
}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
export class RTDetrV2PreTrainedModel extends PreTrainedModel {}
export class RTDetrV2Model extends RTDetrV2PreTrainedModel {}
export class RTDetrV2ForObjectDetection extends RTDetrV2PreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new RTDetrV2ObjectDetectionOutput(await super._call(model_inputs));
    }
}

export class RTDetrV2ObjectDetectionOutput extends RTDetrObjectDetectionOutput {}
//////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
//
// //////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
// ////////////////////////////////////////////////////////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
//
// //////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
// ////////////////////////////////////////////////////////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
//
// //////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
// ////////////////////////////////////////////////////////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
//
// //////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
// ////////////////////////////////////////////////////////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
//
// //////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
// ////////////////////////////////////////////////////////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
//
// //////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
// ////////////////////////////////////////////////////////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
//
// //////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
// ////////////////////////////////////////////////////////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
//
// //////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
// ////////////////////////////////////////////////////////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
//
// //////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
// ////////////////////////////////////////////////////////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
//
// //////////////////////////////////////////////////
//
// //////////////////////////////////////////////////
//
// //////////////////////////////////////////////////

//////////////////////////////////////////////////

//////////////////////////////////////////////////

//////////////////////////////////////////////////

//////////////////////////////////////////////////

//////////////////////////////////////////////////

//////////////////////////////////////////////////


//////////////////////////////////////////////////

//////////////////////////////////////////////////


//////////////////////////////////////////////////

//////////////////////////////////////////////////
// MarianMT models

//////////////////////////////////////////////////

//////////////////////////////////////////////////
// WeSpeakerResNet models


//////////////////////////////////////////////////

//////////////////////////////////////////////////
// VITS models

//////////////////////////////////////////////////



//////////////////////////////////////////////////

//////////////////////////////////////////////////

//////////////////////////////////////////////////

//////////////////////////////////////////////////

//////////////////////////////////////////////////
// Musicgen models


//////////////////////////////////////////////////
// MobileNetV4 models
export class MobileNetV4PreTrainedModel extends PreTrainedModel {}

/**
 * The bare MobileNetV4 model outputting raw hidden-states without any specific head on top.
 */
export class MobileNetV4Model extends MobileNetV4PreTrainedModel {}

/**
 * MobileNetV4 model with an image classification head on top (a linear layer on top of the pooled features),
 * e.g. for ImageNet.
 */
export class MobileNetV4ForImageClassification extends MobileNetV4PreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
export class MobileNetV4ForSemanticSegmentation extends MobileNetV4PreTrainedModel {}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// Decision Transformer models
export class DecisionTransformerPreTrainedModel extends PreTrainedModel {}

/**
 * The model builds upon the GPT2 architecture to perform autoregressive prediction of actions in an offline RL setting.
 * Refer to the paper for more details: https://huggingface.co/papers/2106.01345
 */
export class DecisionTransformerModel extends DecisionTransformerPreTrainedModel {}

//////////////////////////////////////////////////

export class MultiModalityPreTrainedModel extends PreTrainedModel {}
export class MultiModalityCausalLM extends MultiModalityPreTrainedModel {
    forward_params = [
        // prepare_inputs_embeds
        'input_ids',
        'pixel_values',
        'images_seq_mask',
        'images_emb_mask',

        // language_model
        'attention_mask',
        'position_ids',
        'past_key_values',
    ];

    /**
     * @param {ConstructorParameters<typeof MultiModalityPreTrainedModel>} args
     */
    constructor(...args) {
        super(...args);

        // State-based approach to switch out which heads to use during generation
        this._generation_mode = 'text';
    }

    async forward(model_inputs) {
        const mode = this._generation_mode ?? 'text';

        // TODO support re-using PKVs for input_ids.dims[1] !== 1
        // if (model_inputs.past_key_values) {
        //     //  && model_inputs.input_ids.dims[1] === 1
        // }

        let output_1;
        if (mode === 'text' || !model_inputs.past_key_values) {
            const session = this.sessions['prepare_inputs_embeds'];
            const prep_inputs = pick(model_inputs, session.inputNames);
            output_1 = await sessionRun(session, prep_inputs);
        } else {
            const session = this.sessions['gen_img_embeds'];
            const prep_inputs = pick(
                {
                    image_ids: model_inputs.input_ids,
                },
                session.inputNames,
            );
            output_1 = await sessionRun(session, prep_inputs);
        }

        const input_2 = { ...model_inputs, ...output_1 };
        const output_2 = await decoderForward(this, input_2);

        const head = this.sessions[mode === 'text' ? 'lm_head' : 'gen_head'];
        if (!head) {
            throw new Error(`Unable to find "${head}" generation head`);
        }

        const output_3 = await sessionRun(head, pick(output_2, head.inputNames));

        return {
            ...output_1,
            ...output_2,
            ...output_3,
        };
    }

    /**
     * @param {import('./generation/parameters.js').GenerationFunctionParameters} options
     */
    async generate(options) {
        this._generation_mode = 'text';
        return super.generate(options);
    }

    /**
     * @param {import('./generation/parameters.js').GenerationFunctionParameters} options
     */
    async generate_images(options) {
        this._generation_mode = 'image';

        const start_num_tokens = (options.inputs ?? options[this.main_input_name]).dims[1];
        const all_tokens = await super.generate(options);

        const generated_tokens = /** @type {Tensor} */ (all_tokens).slice(null, [start_num_tokens, null]);

        const image_decode = this.sessions['image_decode'];
        const { decoded_image } = await sessionRun(image_decode, {
            generated_tokens,
        });

        // Equivalent to `np.clip((dec + 1) / 2 * 255, 0, 255)`
        const clamped = decoded_image
            .add_(1)
            .mul_(255 / 2)
            .clamp_(0, 255)
            .to('uint8');

        // Return as a list of images
        const images = [];
        for (const tensor of clamped) {
            const img = RawImage.fromTensor(tensor);
            images.push(img);
        }
        return images;
    }
}

export class MgpstrPreTrainedModel extends PreTrainedModel {}

/**
 * MGP-STR Model transformer with three classification heads on top
 * (three A^3 modules and three linear layer on top of the transformer encoder output) for scene text recognition (STR).
 */
export class MgpstrForSceneTextRecognition extends MgpstrPreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new MgpstrModelOutput(await super._call(model_inputs));
    }
}

//////////////////////////////////////////////////
// PatchTST Transformer models
export class PatchTSTPreTrainedModel extends PreTrainedModel {}

/**
 * The bare PatchTST Model outputting raw hidden-states without any specific head.
 */
export class PatchTSTModel extends PatchTSTPreTrainedModel {}

/**
 * The PatchTST for prediction model.
 */
export class PatchTSTForPrediction extends PatchTSTPreTrainedModel {}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// PatchTSMixer Transformer models
export class PatchTSMixerPreTrainedModel extends PreTrainedModel {}

/**
 * The bare PatchTSMixer Model outputting raw hidden-states without any specific head.
 */
export class PatchTSMixerModel extends PatchTSMixerPreTrainedModel {}

/**
 * The PatchTSMixer for prediction model.
 */
export class PatchTSMixerForPrediction extends PatchTSMixerPreTrainedModel {}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
export class UltravoxPreTrainedModel extends PreTrainedModel {
    forward_params = ['input_ids', 'attention_mask', 'position_ids', 'audio_values', 'past_key_values'];
}

export class UltravoxModel extends UltravoxPreTrainedModel {
    _merge_input_ids_with_audio_features(kwargs) {
        const audio_hidden_size = kwargs.audio_features.dims.at(-1);
        const reshaped_audio_features = kwargs.audio_features.view(-1, audio_hidden_size);

        return default_merge_input_ids_with_audio_features({
            // @ts-ignore
            audio_token_id: this.config.ignore_index ?? this.config.audio_token_id,
            ...kwargs,
            audio_features: reshaped_audio_features,
        });
    }
}
//////////////////////////////////////////////////

export class VoxtralForConditionalGeneration extends UltravoxModel {}

//////////////////////////////////////////////////
// Mimi models
export class MimiPreTrainedModel extends PreTrainedModel {
    main_input_name = 'input_values';
    forward_params = ['input_values'];
}

/**
 * The Mimi neural audio codec model.
 */
export class MimiModel extends MimiPreTrainedModel {
    /**
     * Encodes the input audio waveform into discrete codes.
     * @param {Object} inputs Model inputs
     * @param {Tensor} [inputs.input_values] Float values of the input audio waveform, of shape `(batch_size, channels, sequence_length)`).
     * @returns {Promise<MimiEncoderOutput>} The output tensor of shape `(batch_size, num_codebooks, sequence_length)`.
     */
    async encode(inputs) {
        return new MimiEncoderOutput(await sessionRun(this.sessions['encoder_model'], inputs));
    }

    /**
     * Decodes the given frames into an output audio waveform.
     * @param {MimiEncoderOutput} inputs The encoded audio codes.
     * @returns {Promise<MimiDecoderOutput>} The output tensor of shape `(batch_size, num_channels, sequence_length)`.
     */
    async decode(inputs) {
        return new MimiDecoderOutput(await sessionRun(this.sessions['decoder_model'], inputs));
    }
}

export class MimiEncoderModel extends MimiPreTrainedModel {
    /** @type {typeof PreTrainedModel.from_pretrained} */
    static async from_pretrained(pretrained_model_name_or_path, options = {}) {
        return super.from_pretrained(pretrained_model_name_or_path, {
            ...options,
            // Update default model file name if not provided
            model_file_name: options.model_file_name ?? 'encoder_model',
        });
    }
}
export class MimiDecoderModel extends MimiPreTrainedModel {
    /** @type {typeof PreTrainedModel.from_pretrained} */
    static async from_pretrained(pretrained_model_name_or_path, options = {}) {
        return super.from_pretrained(pretrained_model_name_or_path, {
            ...options,
            // Update default model file name if not provided
            model_file_name: options.model_file_name ?? 'decoder_model',
        });
    }
}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// Dac models
export class DacPreTrainedModel extends PreTrainedModel {
    main_input_name = 'input_values';
    forward_params = ['input_values'];
}

/**
 * The DAC (Descript Audio Codec) model.
 */
export class DacModel extends DacPreTrainedModel {
    /**
     * Encodes the input audio waveform into discrete codes.
     * @param {Object} inputs Model inputs
     * @param {Tensor} [inputs.input_values] Float values of the input audio waveform, of shape `(batch_size, channels, sequence_length)`).
     * @returns {Promise<DacEncoderOutput>} The output tensor of shape `(batch_size, num_codebooks, sequence_length)`.
     */
    async encode(inputs) {
        return new DacEncoderOutput(await sessionRun(this.sessions['encoder_model'], inputs));
    }

    /**
     * Decodes the given frames into an output audio waveform.
     * @param {DacEncoderOutput} inputs The encoded audio codes.
     * @returns {Promise<DacDecoderOutput>} The output tensor of shape `(batch_size, num_channels, sequence_length)`.
     */
    async decode(inputs) {
        return new DacDecoderOutput(await sessionRun(this.sessions['decoder_model'], inputs));
    }
}

export class DacEncoderModel extends DacPreTrainedModel {
    /** @type {typeof PreTrainedModel.from_pretrained} */
    static async from_pretrained(pretrained_model_name_or_path, options = {}) {
        return super.from_pretrained(pretrained_model_name_or_path, {
            ...options,
            // Update default model file name if not provided
            model_file_name: options.model_file_name ?? 'encoder_model',
        });
    }
}
export class DacDecoderModel extends DacPreTrainedModel {
    /** @type {typeof PreTrainedModel.from_pretrained} */
    static async from_pretrained(pretrained_model_name_or_path, options = {}) {
        return super.from_pretrained(pretrained_model_name_or_path, {
            ...options,
            // Update default model file name if not provided
            model_file_name: options.model_file_name ?? 'decoder_model',
        });
    }
}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// Snac models
export class SnacPreTrainedModel extends PreTrainedModel {
    main_input_name = 'input_values';
    forward_params = ['input_values'];
}

/**
 * The SNAC (Multi-Scale Neural Audio Codec) model.
 */
export class SnacModel extends SnacPreTrainedModel {
    /**
     * Encodes the input audio waveform into discrete codes.
     * @param {Object} inputs Model inputs
     * @param {Tensor} [inputs.input_values] Float values of the input audio waveform, of shape `(batch_size, channels, sequence_length)`).
     * @returns {Promise<Record<string, Tensor>>} The output tensors of shape `(batch_size, num_codebooks, sequence_length)`.
     */
    async encode(inputs) {
        return await sessionRun(this.sessions['encoder_model'], inputs);
    }

    /**
     * Decodes the given frames into an output audio waveform.
     * @param {Record<string, Tensor>} inputs The encoded audio codes.
     * @returns {Promise<{audio_values: Tensor}>} The output tensor of shape `(batch_size, num_channels, sequence_length)`.
     */
    async decode(inputs) {
        return await sessionRun(this.sessions['decoder_model'], inputs);
    }
}

export class SnacEncoderModel extends SnacPreTrainedModel {
    /** @type {typeof PreTrainedModel.from_pretrained} */
    static async from_pretrained(pretrained_model_name_or_path, options = {}) {
        return super.from_pretrained(pretrained_model_name_or_path, {
            ...options,
            // Update default model file name if not provided
            model_file_name: options.model_file_name ?? 'encoder_model',
        });
    }
}
export class SnacDecoderModel extends SnacPreTrainedModel {
    /** @type {typeof PreTrainedModel.from_pretrained} */
    static async from_pretrained(pretrained_model_name_or_path, options = {}) {
        return super.from_pretrained(pretrained_model_name_or_path, {
            ...options,
            // Update default model file name if not provided
            model_file_name: options.model_file_name ?? 'decoder_model',
        });
    }
}
//////////////////////////////////////////////////

export class ChatterboxPreTrainedModel extends PreTrainedModel {
    forward_params = [
        'input_ids',
        'inputs_embeds',
        'attention_mask',
        'position_ids',
        'audio_values',
        'exaggeration',
        'audio_features',
        'audio_tokens',
        'speaker_embeddings',
        'speaker_features',
        'past_key_values',
    ];
    main_input_name = 'input_ids';

    _return_dict_in_generate_keys = ['audio_tokens', 'speaker_embeddings', 'speaker_features'];
}
export class ChatterboxModel extends ChatterboxPreTrainedModel {
    /**
     * @param {Tensor} audio_values
     * @returns {Promise<{audio_features: Tensor, audio_tokens: Tensor, speaker_embeddings: Tensor, speaker_features: Tensor}>}
     */
    async encode_speech(audio_values) {
        return sessionRun(this.sessions['speech_encoder'], {
            audio_values,
        });
    }

    async forward({
        // Produced by the tokenizer/processor:
        input_ids = null,
        attention_mask = null,
        audio_values = null,
        exaggeration = null,

        // Used during generation:
        position_ids = null,
        inputs_embeds = null,
        past_key_values = null,

        // Generic generation parameters
        generation_config = null,
        logits_processor = null,

        // Speaker embeddings/features (useful for re-using pre-computed speaker data)
        audio_features = null, // float32[batch_size,sequence_length,1024]
        audio_tokens = null, // int64[batch_size,audio_sequence_length]
        speaker_embeddings = null, // float32[batch_size,192]
        speaker_features = null, // float32[batch_size,feature_dim,80]

        // TODO: needed?
        ...kwargs
    }) {
        let speech_encoder_outputs;
        if (!inputs_embeds) {
            const expected_inputs = this.sessions['embed_tokens'].inputNames;
            const embed_model_inputs = { input_ids };
            if (expected_inputs.includes('exaggeration')) {
                // Support the following types for exaggeration:
                // 1. null/undefined (no exaggeration): use the default of 0.5
                // 2. number: broadcast to (batch_size,)
                // 3. number[]: convert to Tensor of shape (batch_size,)
                // 4. Tensor of shape (batch_size, 1)
                if (!(exaggeration instanceof Tensor)) {
                    const batch_size = input_ids.dims[0];
                    if (exaggeration == null) {
                        exaggeration = full([batch_size], 0.5);
                    } else if (typeof exaggeration === 'number') {
                        exaggeration = full([batch_size], exaggeration);
                    } else if (Array.isArray(exaggeration)) {
                        exaggeration = new Tensor('float32', exaggeration, [batch_size]);
                    } else {
                        throw new Error('Unsupported type for `exaggeration` input');
                    }
                }
                embed_model_inputs.exaggeration = exaggeration;
            }
            if (expected_inputs.includes('position_ids')) {
                embed_model_inputs.position_ids = position_ids;
            }

            ({ inputs_embeds } = await sessionRun(this.sessions['embed_tokens'], embed_model_inputs));

            if (audio_features && audio_tokens && speaker_embeddings && speaker_features) {
                // Use pre-computed speech encoder outputs
                speech_encoder_outputs = { audio_features, audio_tokens, speaker_embeddings, speaker_features };
            }

            if (speech_encoder_outputs || audio_values) {
                speech_encoder_outputs ??= await this.encode_speech(audio_values);

                // Update LLM inputs
                inputs_embeds = cat([speech_encoder_outputs.audio_features, inputs_embeds], 1);
                attention_mask = ones([inputs_embeds.dims[0], inputs_embeds.dims[1]]);
            } else {
                const target_length = inputs_embeds.dims[1];
                if (!past_key_values || target_length !== 1) {
                    throw new Error('Incorrect state encountered during generation.');
                }
                const past_length = Object.values(past_key_values)[0].dims.at(-2);
                attention_mask = ones([inputs_embeds.dims[0], past_length + target_length]);
            }
        }

        const outputs = await decoderForward(
            this,
            {
                inputs_embeds,
                past_key_values,
                attention_mask,
                generation_config,
                logits_processor,
            },
            false,
        );
        return {
            ...outputs,
            ...speech_encoder_outputs,
        };
    }

    /** @type {PreTrainedModel['generate']} */
    async generate(params) {
        const { sequences, audio_tokens, speaker_embeddings, speaker_features } = /** @type {any} */ (
            await super.generate({
                ...params,
                return_dict_in_generate: true,
            })
        );

        const new_tokens = sequences.slice(null, [
            params.input_ids.dims[1], // Exclude start of speech token
            -1, // Exclude end of speech token
        ]);

        const SILENCE_TOKEN = 4299n;
        const silence_tokens = full([new_tokens.dims[0], 3], SILENCE_TOKEN); // Add 3 silence tokens
        const speech_tokens = cat([audio_tokens, new_tokens, silence_tokens], 1);

        const { waveform } = await sessionRun(this.sessions['conditional_decoder'], {
            speech_tokens,
            speaker_features,
            speaker_embeddings,
        });
        return waveform;
    }
}

//////////////////////////////////////////////////
// AutoModels, used to simplify construction of PreTrainedModels
// (uses config to instantiate correct class)

/**
 * Base class of all AutoModels. Contains the `from_pretrained` function
 * which is used to instantiate pretrained models.
 */
export class PretrainedMixin {
    /**
     * Mapping from model type to model class.
     * @type {Map<string, Object>[]}
     */
    static MODEL_CLASS_MAPPINGS = null;

    /**
     * Whether to attempt to instantiate the base class (`PretrainedModel`) if
     * the model type is not found in the mapping.
     */
    static BASE_IF_FAIL = false;

    /** @type {typeof PreTrainedModel.from_pretrained} */
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
        options.config = await AutoConfig.from_pretrained(pretrained_model_name_or_path, options);

        if (!this.MODEL_CLASS_MAPPINGS) {
            throw new Error('`MODEL_CLASS_MAPPINGS` not implemented for this type of `AutoClass`: ' + this.name);
        }
        const model_type = options.config.model_type;
        for (const MODEL_CLASS_MAPPING of this.MODEL_CLASS_MAPPINGS) {
            let modelInfo = MODEL_CLASS_MAPPING.get(model_type);
            if (!modelInfo) {
                // As a fallback, we check if model_type is specified as the exact class
                for (const cls of MODEL_CLASS_MAPPING.values()) {
                    if (cls[0] === model_type) {
                        modelInfo = cls;
                        break;
                    }
                }
                if (!modelInfo) continue; // Item not found in this mapping
            }
            return await modelInfo[1].from_pretrained(pretrained_model_name_or_path, options);
        }

        if (this.BASE_IF_FAIL) {
            if (!CUSTOM_ARCHITECTURES.has(model_type)) {
                console.warn(`Unknown model class "${model_type}", attempting to construct from base class.`);
            }
            return await PreTrainedModel.from_pretrained(pretrained_model_name_or_path, options);
        } else {
            throw Error(`Unsupported model type: ${model_type}`);
        }
    }
}

const MODEL_MAPPING_NAMES_ENCODER_ONLY = new Map([
    ['bert', ['BertModel', BertModel]],
    ['neobert', ['NeoBertModel', NeoBertModel]],
    ['modernbert', ['ModernBertModel', ModernBertModel]],
    ['nomic_bert', ['NomicBertModel', NomicBertModel]],
    ['roformer', ['RoFormerModel', RoFormerModel]],
    ['electra', ['ElectraModel', ElectraModel]],
    ['esm', ['EsmModel', EsmModel]],
    ['convbert', ['ConvBertModel', ConvBertModel]],
    ['camembert', ['CamembertModel', CamembertModel]],
    ['deberta', ['DebertaModel', DebertaModel]],
    ['deberta-v2', ['DebertaV2Model', DebertaV2Model]],
    ['mpnet', ['MPNetModel', MPNetModel]],
    ['albert', ['AlbertModel', AlbertModel]],
    ['distilbert', ['DistilBertModel', DistilBertModel]],
    ['roberta', ['RobertaModel', RobertaModel]],
    ['xlm', ['XLMModel', XLMModel]],
    ['xlm-roberta', ['XLMRobertaModel', XLMRobertaModel]],
    ['clap', ['ClapModel', ClapModel]],
    ['clip', ['CLIPModel', CLIPModel]],
    ['clipseg', ['CLIPSegModel', CLIPSegModel]],
    ['chinese_clip', ['ChineseCLIPModel', ChineseCLIPModel]],
    ['siglip', ['SiglipModel', SiglipModel]],
    ['jina_clip', ['JinaCLIPModel', JinaCLIPModel]],
    ['mobilebert', ['MobileBertModel', MobileBertModel]],
    ['squeezebert', ['SqueezeBertModel', SqueezeBertModel]],
    ['wav2vec2', ['Wav2Vec2Model', Wav2Vec2Model]],
    ['wav2vec2-bert', ['Wav2Vec2BertModel', Wav2Vec2BertModel]],
    ['unispeech', ['UniSpeechModel', UniSpeechModel]],
    ['unispeech-sat', ['UniSpeechSatModel', UniSpeechSatModel]],
    ['hubert', ['HubertModel', HubertModel]],
    ['wavlm', ['WavLMModel', WavLMModel]],
    ['audio-spectrogram-transformer', ['ASTModel', ASTModel]],
    ['vits', ['VitsModel', VitsModel]],
    ['pyannote', ['PyAnnoteModel', PyAnnoteModel]],
    ['wespeaker-resnet', ['WeSpeakerResNetModel', WeSpeakerResNetModel]],

    ['detr', ['DetrModel', DetrModel]],
    ['rt_detr', ['RTDetrModel', RTDetrModel]],
    ['rt_detr_v2', ['RTDetrV2Model', RTDetrV2Model]],
    ['rf_detr', ['RFDetrModel', RFDetrModel]],
    ['d_fine', ['DFineModel', DFineModel]],
    ['table-transformer', ['TableTransformerModel', TableTransformerModel]],
    ['vit', ['ViTModel', ViTModel]],
    ['ijepa', ['IJepaModel', IJepaModel]],
    ['pvt', ['PvtModel', PvtModel]],
    ['vit_msn', ['ViTMSNModel', ViTMSNModel]],
    ['vit_mae', ['ViTMAEModel', ViTMAEModel]],
    ['groupvit', ['GroupViTModel', GroupViTModel]],
    ['fastvit', ['FastViTModel', FastViTModel]],
    ['mobilevit', ['MobileViTModel', MobileViTModel]],
    ['mobilevitv2', ['MobileViTV2Model', MobileViTV2Model]],
    ['owlvit', ['OwlViTModel', OwlViTModel]],
    ['owlv2', ['Owlv2Model', Owlv2Model]],
    ['beit', ['BeitModel', BeitModel]],
    ['deit', ['DeiTModel', DeiTModel]],
    ['hiera', ['HieraModel', HieraModel]],
    ['convnext', ['ConvNextModel', ConvNextModel]],
    ['convnextv2', ['ConvNextV2Model', ConvNextV2Model]],
    ['dinov2', ['Dinov2Model', Dinov2Model]],
    ['dinov2_with_registers', ['Dinov2WithRegistersModel', Dinov2WithRegistersModel]],
    ['dinov3_vit', ['DINOv3ViTModel', DINOv3ViTModel]],
    ['dinov3_convnext', ['DINOv3ConvNextModel', DINOv3ConvNextModel]],
    ['resnet', ['ResNetModel', ResNetModel]],
    ['swin', ['SwinModel', SwinModel]],
    ['swin2sr', ['Swin2SRModel', Swin2SRModel]],
    ['donut-swin', ['DonutSwinModel', DonutSwinModel]],
    ['yolos', ['YolosModel', YolosModel]],
    ['dpt', ['DPTModel', DPTModel]],
    ['glpn', ['GLPNModel', GLPNModel]],

    ['hifigan', ['SpeechT5HifiGan', SpeechT5HifiGan]],
    ['efficientnet', ['EfficientNetModel', EfficientNetModel]],

    ['decision_transformer', ['DecisionTransformerModel', DecisionTransformerModel]],
    ['patchtst', ['PatchTSTForPrediction', PatchTSTModel]],
    ['patchtsmixer', ['PatchTSMixerForPrediction', PatchTSMixerModel]],

    ['mobilenet_v1', ['MobileNetV1Model', MobileNetV1Model]],
    ['mobilenet_v2', ['MobileNetV2Model', MobileNetV2Model]],
    ['mobilenet_v3', ['MobileNetV3Model', MobileNetV3Model]],
    ['mobilenet_v4', ['MobileNetV4Model', MobileNetV4Model]],

    ['maskformer', ['MaskFormerModel', MaskFormerModel]],
    ['mgp-str', ['MgpstrForSceneTextRecognition', MgpstrForSceneTextRecognition]],

    ['style_text_to_speech_2', ['StyleTextToSpeech2Model', StyleTextToSpeech2Model]],
]);

const MODEL_MAPPING_NAMES_ENCODER_DECODER = new Map([
    ['t5', ['T5Model', T5Model]],
    ['longt5', ['LongT5Model', LongT5Model]],
    ['mt5', ['MT5Model', MT5Model]],
    ['bart', ['BartModel', BartModel]],
    ['mbart', ['MBartModel', MBartModel]],
    ['marian', ['MarianModel', MarianModel]],
    ['whisper', ['WhisperModel', WhisperModel]],
    ['m2m_100', ['M2M100Model', M2M100Model]],
    ['blenderbot', ['BlenderbotModel', BlenderbotModel]],
    ['blenderbot-small', ['BlenderbotSmallModel', BlenderbotSmallModel]],
]);

const MODEL_MAPPING_NAMES_AUTO_ENCODER = new Map([
    ['mimi', ['MimiModel', MimiModel]],
    ['dac', ['DacModel', DacModel]],
    ['snac', ['SnacModel', SnacModel]],
]);

const MODEL_MAPPING_NAMES_DECODER_ONLY = new Map([
    ['bloom', ['BloomModel', BloomModel]],
    ['jais', ['JAISModel', JAISModel]],
    ['gpt2', ['GPT2Model', GPT2Model]],
    ['gpt_oss', ['GptOssModel', GptOssModel]],
    ['gptj', ['GPTJModel', GPTJModel]],
    ['gpt_bigcode', ['GPTBigCodeModel', GPTBigCodeModel]],
    ['gpt_neo', ['GPTNeoModel', GPTNeoModel]],
    ['gpt_neox', ['GPTNeoXModel', GPTNeoXModel]],
    ['codegen', ['CodeGenModel', CodeGenModel]],
    ['llama', ['LlamaModel', LlamaModel]],
    ['apertus', ['ApertusModel', ApertusModel]],
    ['nanochat', ['NanoChatModel', NanoChatModel]],
    ['arcee', ['ArceeModel', ArceeModel]],
    ['lfm2', ['Lfm2Model', Lfm2Model]],
    ['smollm3', ['SmolLM3Model', SmolLM3Model]],
    ['exaone', ['ExaoneModel', ExaoneModel]],
    ['olmo', ['OlmoModel', OlmoModel]],
    ['olmo2', ['Olmo2Model', Olmo2Model]],
    ['olmo3', ['Olmo3Model', Olmo3Model]],
    ['mobilellm', ['MobileLLMModel', MobileLLMModel]],
    ['granite', ['GraniteModel', GraniteModel]],
    ['granitemoehybrid', ['GraniteMoeHybridModel', GraniteMoeHybridModel]],
    ['cohere', ['CohereModel', CohereModel]],
    ['gemma', ['GemmaModel', GemmaModel]],
    ['gemma2', ['Gemma2Model', Gemma2Model]],
    ['vaultgemma', ['VaultGemmaModel', VaultGemmaModel]],
    ['gemma3_text', ['Gemma3Model', Gemma3Model]],
    ['helium', ['HeliumModel', HeliumModel]],
    ['glm', ['GlmModel', GlmModel]],
    ['openelm', ['OpenELMModel', OpenELMModel]],
    ['qwen2', ['Qwen2Model', Qwen2Model]],
    ['qwen3', ['Qwen3Model', Qwen3Model]],
    ['phi', ['PhiModel', PhiModel]],
    ['phi3', ['Phi3Model', Phi3Model]],
    ['mpt', ['MptModel', MptModel]],
    ['opt', ['OPTModel', OPTModel]],
    ['mistral', ['MistralModel', MistralModel]],
    ['ernie4_5', ['Ernie4_5_Model', Ernie4_5_Model]],
    ['starcoder2', ['Starcoder2Model', Starcoder2Model]],
    ['falcon', ['FalconModel', FalconModel]],
    ['stablelm', ['StableLmModel', StableLmModel]],
    ['modernbert-decoder', ['ModernBertDecoderModel', ModernBertDecoderModel]],
]);

const MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES = new Map([
    ['speecht5', ['SpeechT5ForSpeechToText', SpeechT5ForSpeechToText]],
    ['whisper', ['WhisperForConditionalGeneration', WhisperForConditionalGeneration]],
    ['lite-whisper', ['LiteWhisperForConditionalGeneration', LiteWhisperForConditionalGeneration]],
    ['moonshine', ['MoonshineForConditionalGeneration', MoonshineForConditionalGeneration]],
]);

const MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES = new Map([
    ['speecht5', ['SpeechT5ForTextToSpeech', SpeechT5ForTextToSpeech]],
]);

const MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES = new Map([
    ['vits', ['VitsModel', VitsModel]],
    ['musicgen', ['MusicgenForConditionalGeneration', MusicgenForConditionalGeneration]],
    ['supertonic', ['SupertonicForConditionalGeneration', SupertonicForConditionalGeneration]],
]);

const MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = new Map([
    ['bert', ['BertForSequenceClassification', BertForSequenceClassification]],
    ['neobert', ['NeoBertForSequenceClassification', NeoBertForSequenceClassification]],
    ['modernbert', ['ModernBertForSequenceClassification', ModernBertForSequenceClassification]],
    ['roformer', ['RoFormerForSequenceClassification', RoFormerForSequenceClassification]],
    ['electra', ['ElectraForSequenceClassification', ElectraForSequenceClassification]],
    ['esm', ['EsmForSequenceClassification', EsmForSequenceClassification]],
    ['convbert', ['ConvBertForSequenceClassification', ConvBertForSequenceClassification]],
    ['camembert', ['CamembertForSequenceClassification', CamembertForSequenceClassification]],
    ['deberta', ['DebertaForSequenceClassification', DebertaForSequenceClassification]],
    ['deberta-v2', ['DebertaV2ForSequenceClassification', DebertaV2ForSequenceClassification]],
    ['mpnet', ['MPNetForSequenceClassification', MPNetForSequenceClassification]],
    ['albert', ['AlbertForSequenceClassification', AlbertForSequenceClassification]],
    ['distilbert', ['DistilBertForSequenceClassification', DistilBertForSequenceClassification]],
    ['roberta', ['RobertaForSequenceClassification', RobertaForSequenceClassification]],
    ['xlm', ['XLMForSequenceClassification', XLMForSequenceClassification]],
    ['xlm-roberta', ['XLMRobertaForSequenceClassification', XLMRobertaForSequenceClassification]],
    ['bart', ['BartForSequenceClassification', BartForSequenceClassification]],
    ['mbart', ['MBartForSequenceClassification', MBartForSequenceClassification]],
    ['mobilebert', ['MobileBertForSequenceClassification', MobileBertForSequenceClassification]],
    ['squeezebert', ['SqueezeBertForSequenceClassification', SqueezeBertForSequenceClassification]],
]);

const MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES = new Map([
    ['bert', ['BertForTokenClassification', BertForTokenClassification]],
    ['neobert', ['NeoBertForTokenClassification', NeoBertForTokenClassification]],
    ['modernbert', ['ModernBertForTokenClassification', ModernBertForTokenClassification]],
    ['roformer', ['RoFormerForTokenClassification', RoFormerForTokenClassification]],
    ['electra', ['ElectraForTokenClassification', ElectraForTokenClassification]],
    ['esm', ['EsmForTokenClassification', EsmForTokenClassification]],
    ['convbert', ['ConvBertForTokenClassification', ConvBertForTokenClassification]],
    ['camembert', ['CamembertForTokenClassification', CamembertForTokenClassification]],
    ['deberta', ['DebertaForTokenClassification', DebertaForTokenClassification]],
    ['deberta-v2', ['DebertaV2ForTokenClassification', DebertaV2ForTokenClassification]],
    ['mpnet', ['MPNetForTokenClassification', MPNetForTokenClassification]],
    ['distilbert', ['DistilBertForTokenClassification', DistilBertForTokenClassification]],
    ['roberta', ['RobertaForTokenClassification', RobertaForTokenClassification]],
    ['xlm', ['XLMForTokenClassification', XLMForTokenClassification]],
    ['xlm-roberta', ['XLMRobertaForTokenClassification', XLMRobertaForTokenClassification]],
]);

const MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES = new Map([
    ['t5', ['T5ForConditionalGeneration', T5ForConditionalGeneration]],
    ['longt5', ['LongT5ForConditionalGeneration', LongT5ForConditionalGeneration]],
    ['mt5', ['MT5ForConditionalGeneration', MT5ForConditionalGeneration]],
    ['bart', ['BartForConditionalGeneration', BartForConditionalGeneration]],
    ['mbart', ['MBartForConditionalGeneration', MBartForConditionalGeneration]],
    ['marian', ['MarianMTModel', MarianMTModel]],
    ['m2m_100', ['M2M100ForConditionalGeneration', M2M100ForConditionalGeneration]],
    ['blenderbot', ['BlenderbotForConditionalGeneration', BlenderbotForConditionalGeneration]],
    ['blenderbot-small', ['BlenderbotSmallForConditionalGeneration', BlenderbotSmallForConditionalGeneration]],
]);

const MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = new Map([
    ['bloom', ['BloomForCausalLM', BloomForCausalLM]],
    ['gpt2', ['GPT2LMHeadModel', GPT2LMHeadModel]],
    ['gpt_oss', ['GptOssForCausalLM', GptOssForCausalLM]],
    ['jais', ['JAISLMHeadModel', JAISLMHeadModel]],
    ['gptj', ['GPTJForCausalLM', GPTJForCausalLM]],
    ['gpt_bigcode', ['GPTBigCodeForCausalLM', GPTBigCodeForCausalLM]],
    ['gpt_neo', ['GPTNeoForCausalLM', GPTNeoForCausalLM]],
    ['gpt_neox', ['GPTNeoXForCausalLM', GPTNeoXForCausalLM]],
    ['codegen', ['CodeGenForCausalLM', CodeGenForCausalLM]],
    ['llama', ['LlamaForCausalLM', LlamaForCausalLM]],
    ['nanochat', ['NanoChatForCausalLM', NanoChatForCausalLM]],
    ['apertus', ['ApertusForCausalLM', ApertusForCausalLM]],
    ['llama4_text', ['Llama4ForCausalLM', Llama4ForCausalLM]],
    ['arcee', ['ArceeForCausalLM', ArceeForCausalLM]],
    ['lfm2', ['Lfm2ForCausalLM', Lfm2ForCausalLM]],
    ['smollm3', ['SmolLM3ForCausalLM', SmolLM3ForCausalLM]],
    ['exaone', ['ExaoneForCausalLM', ExaoneForCausalLM]],
    ['olmo', ['OlmoForCausalLM', OlmoForCausalLM]],
    ['olmo2', ['Olmo2ForCausalLM', Olmo2ForCausalLM]],
    ['olmo3', ['Olmo3ForCausalLM', Olmo3ForCausalLM]],
    ['mobilellm', ['MobileLLMForCausalLM', MobileLLMForCausalLM]],
    ['granite', ['GraniteForCausalLM', GraniteForCausalLM]],
    ['granitemoehybrid', ['GraniteMoeHybridForCausalLM', GraniteMoeHybridForCausalLM]],
    ['cohere', ['CohereForCausalLM', CohereForCausalLM]],
    ['gemma', ['GemmaForCausalLM', GemmaForCausalLM]],
    ['gemma2', ['Gemma2ForCausalLM', Gemma2ForCausalLM]],
    ['vaultgemma', ['VaultGemmaForCausalLM', VaultGemmaForCausalLM]],
    ['gemma3_text', ['Gemma3ForCausalLM', Gemma3ForCausalLM]],
    ['helium', ['HeliumForCausalLM', HeliumForCausalLM]],
    ['glm', ['GlmForCausalLM', GlmForCausalLM]],
    ['openelm', ['OpenELMForCausalLM', OpenELMForCausalLM]],
    ['qwen2', ['Qwen2ForCausalLM', Qwen2ForCausalLM]],
    ['qwen3', ['Qwen3ForCausalLM', Qwen3ForCausalLM]],
    ['phi', ['PhiForCausalLM', PhiForCausalLM]],
    ['phi3', ['Phi3ForCausalLM', Phi3ForCausalLM]],
    ['mpt', ['MptForCausalLM', MptForCausalLM]],
    ['opt', ['OPTForCausalLM', OPTForCausalLM]],
    ['mbart', ['MBartForCausalLM', MBartForCausalLM]],
    ['mistral', ['MistralForCausalLM', MistralForCausalLM]],
    ['ernie4_5', ['Ernie4_5_ForCausalLM', Ernie4_5_ForCausalLM]],
    ['starcoder2', ['Starcoder2ForCausalLM', Starcoder2ForCausalLM]],
    ['falcon', ['FalconForCausalLM', FalconForCausalLM]],
    ['trocr', ['TrOCRForCausalLM', TrOCRForCausalLM]],
    ['stablelm', ['StableLmForCausalLM', StableLmForCausalLM]],
    ['modernbert-decoder', ['ModernBertDecoderForCausalLM', ModernBertDecoderForCausalLM]],

    // Also image-text-to-text
    ['phi3_v', ['Phi3VForCausalLM', Phi3VForCausalLM]],
]);

const MODEL_FOR_MULTIMODALITY_MAPPING_NAMES = new Map([
    ['multi_modality', ['MultiModalityCausalLM', MultiModalityCausalLM]],
]);

const MODEL_FOR_MASKED_LM_MAPPING_NAMES = new Map([
    ['bert', ['BertForMaskedLM', BertForMaskedLM]],
    ['neobert', ['NeoBertForMaskedLM', NeoBertForMaskedLM]],
    ['modernbert', ['ModernBertForMaskedLM', ModernBertForMaskedLM]],
    ['roformer', ['RoFormerForMaskedLM', RoFormerForMaskedLM]],
    ['electra', ['ElectraForMaskedLM', ElectraForMaskedLM]],
    ['esm', ['EsmForMaskedLM', EsmForMaskedLM]],
    ['convbert', ['ConvBertForMaskedLM', ConvBertForMaskedLM]],
    ['camembert', ['CamembertForMaskedLM', CamembertForMaskedLM]],
    ['deberta', ['DebertaForMaskedLM', DebertaForMaskedLM]],
    ['deberta-v2', ['DebertaV2ForMaskedLM', DebertaV2ForMaskedLM]],
    ['mpnet', ['MPNetForMaskedLM', MPNetForMaskedLM]],
    ['albert', ['AlbertForMaskedLM', AlbertForMaskedLM]],
    ['distilbert', ['DistilBertForMaskedLM', DistilBertForMaskedLM]],
    ['roberta', ['RobertaForMaskedLM', RobertaForMaskedLM]],
    ['xlm', ['XLMWithLMHeadModel', XLMWithLMHeadModel]],
    ['xlm-roberta', ['XLMRobertaForMaskedLM', XLMRobertaForMaskedLM]],
    ['mobilebert', ['MobileBertForMaskedLM', MobileBertForMaskedLM]],
    ['squeezebert', ['SqueezeBertForMaskedLM', SqueezeBertForMaskedLM]],
]);

const MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = new Map([
    ['bert', ['BertForQuestionAnswering', BertForQuestionAnswering]],
    ['neobert', ['NeoBertForQuestionAnswering', NeoBertForQuestionAnswering]],
    ['roformer', ['RoFormerForQuestionAnswering', RoFormerForQuestionAnswering]],
    ['electra', ['ElectraForQuestionAnswering', ElectraForQuestionAnswering]],
    ['convbert', ['ConvBertForQuestionAnswering', ConvBertForQuestionAnswering]],
    ['camembert', ['CamembertForQuestionAnswering', CamembertForQuestionAnswering]],
    ['deberta', ['DebertaForQuestionAnswering', DebertaForQuestionAnswering]],
    ['deberta-v2', ['DebertaV2ForQuestionAnswering', DebertaV2ForQuestionAnswering]],
    ['mpnet', ['MPNetForQuestionAnswering', MPNetForQuestionAnswering]],
    ['albert', ['AlbertForQuestionAnswering', AlbertForQuestionAnswering]],
    ['distilbert', ['DistilBertForQuestionAnswering', DistilBertForQuestionAnswering]],
    ['roberta', ['RobertaForQuestionAnswering', RobertaForQuestionAnswering]],
    ['xlm', ['XLMForQuestionAnswering', XLMForQuestionAnswering]],
    ['xlm-roberta', ['XLMRobertaForQuestionAnswering', XLMRobertaForQuestionAnswering]],
    ['mobilebert', ['MobileBertForQuestionAnswering', MobileBertForQuestionAnswering]],
    ['squeezebert', ['SqueezeBertForQuestionAnswering', SqueezeBertForQuestionAnswering]],
]);

const MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = new Map([
    ['vision-encoder-decoder', ['VisionEncoderDecoderModel', VisionEncoderDecoderModel]],
    ['idefics3', ['Idefics3ForConditionalGeneration', Idefics3ForConditionalGeneration]],
    ['smolvlm', ['SmolVLMForConditionalGeneration', SmolVLMForConditionalGeneration]],
]);

const MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES = new Map([
    ['llava', ['LlavaForConditionalGeneration', LlavaForConditionalGeneration]],
    ['llava_onevision', ['LlavaOnevisionForConditionalGeneration', LlavaOnevisionForConditionalGeneration]],
    ['moondream1', ['Moondream1ForConditionalGeneration', Moondream1ForConditionalGeneration]],
    ['florence2', ['Florence2ForConditionalGeneration', Florence2ForConditionalGeneration]],
    ['qwen2-vl', ['Qwen2VLForConditionalGeneration', Qwen2VLForConditionalGeneration]],
    ['idefics3', ['Idefics3ForConditionalGeneration', Idefics3ForConditionalGeneration]],
    ['smolvlm', ['SmolVLMForConditionalGeneration', SmolVLMForConditionalGeneration]],
    ['paligemma', ['PaliGemmaForConditionalGeneration', PaliGemmaForConditionalGeneration]],
    ['llava_qwen2', ['LlavaQwen2ForCausalLM', LlavaQwen2ForCausalLM]],
    ['gemma3n', ['Gemma3nForConditionalGeneration', Gemma3nForConditionalGeneration]],
]);

const MODEL_FOR_AUDIO_TEXT_TO_TEXT_MAPPING_NAMES = new Map([
    ['ultravox', ['UltravoxModel', UltravoxModel]],
    ['voxtral', ['VoxtralForConditionalGeneration', VoxtralForConditionalGeneration]],
]);

const MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES = new Map([
    ['vision-encoder-decoder', ['VisionEncoderDecoderModel', VisionEncoderDecoderModel]],
]);

const MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES = new Map([
    ['vit', ['ViTForImageClassification', ViTForImageClassification]],
    ['ijepa', ['IJepaForImageClassification', IJepaForImageClassification]],
    ['pvt', ['PvtForImageClassification', PvtForImageClassification]],
    ['vit_msn', ['ViTMSNForImageClassification', ViTMSNForImageClassification]],
    ['fastvit', ['FastViTForImageClassification', FastViTForImageClassification]],
    ['mobilevit', ['MobileViTForImageClassification', MobileViTForImageClassification]],
    ['mobilevitv2', ['MobileViTV2ForImageClassification', MobileViTV2ForImageClassification]],
    ['beit', ['BeitForImageClassification', BeitForImageClassification]],
    ['deit', ['DeiTForImageClassification', DeiTForImageClassification]],
    ['hiera', ['HieraForImageClassification', HieraForImageClassification]],
    ['convnext', ['ConvNextForImageClassification', ConvNextForImageClassification]],
    ['convnextv2', ['ConvNextV2ForImageClassification', ConvNextV2ForImageClassification]],
    ['dinov2', ['Dinov2ForImageClassification', Dinov2ForImageClassification]],
    ['dinov2_with_registers', ['Dinov2WithRegistersForImageClassification', Dinov2WithRegistersForImageClassification]],
    ['resnet', ['ResNetForImageClassification', ResNetForImageClassification]],
    ['swin', ['SwinForImageClassification', SwinForImageClassification]],
    ['segformer', ['SegformerForImageClassification', SegformerForImageClassification]],
    ['efficientnet', ['EfficientNetForImageClassification', EfficientNetForImageClassification]],
    ['mobilenet_v1', ['MobileNetV1ForImageClassification', MobileNetV1ForImageClassification]],
    ['mobilenet_v2', ['MobileNetV2ForImageClassification', MobileNetV2ForImageClassification]],
    ['mobilenet_v3', ['MobileNetV3ForImageClassification', MobileNetV3ForImageClassification]],
    ['mobilenet_v4', ['MobileNetV4ForImageClassification', MobileNetV4ForImageClassification]],
]);

const MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES = new Map([
    ['detr', ['DetrForObjectDetection', DetrForObjectDetection]],
    ['rt_detr', ['RTDetrForObjectDetection', RTDetrForObjectDetection]],
    ['rt_detr_v2', ['RTDetrV2ForObjectDetection', RTDetrV2ForObjectDetection]],
    ['rf_detr', ['RFDetrForObjectDetection', RFDetrForObjectDetection]],
    ['d_fine', ['DFineForObjectDetection', DFineForObjectDetection]],
    ['table-transformer', ['TableTransformerForObjectDetection', TableTransformerForObjectDetection]],
    ['yolos', ['YolosForObjectDetection', YolosForObjectDetection]],
]);

const MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES = new Map([
    ['owlvit', ['OwlViTForObjectDetection', OwlViTForObjectDetection]],
    ['owlv2', ['Owlv2ForObjectDetection', Owlv2ForObjectDetection]],
    ['grounding-dino', ['GroundingDinoForObjectDetection', GroundingDinoForObjectDetection]],
]);

const MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES = new Map([
    // TODO: Do not add new models here
    ['detr', ['DetrForSegmentation', DetrForSegmentation]],
    ['clipseg', ['CLIPSegForImageSegmentation', CLIPSegForImageSegmentation]],
]);

const MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES = new Map([
    ['segformer', ['SegformerForSemanticSegmentation', SegformerForSemanticSegmentation]],
    ['sapiens', ['SapiensForSemanticSegmentation', SapiensForSemanticSegmentation]],

    ['swin', ['SwinForSemanticSegmentation', SwinForSemanticSegmentation]],
    ['mobilenet_v1', ['MobileNetV1ForSemanticSegmentation', MobileNetV1ForSemanticSegmentation]],
    ['mobilenet_v2', ['MobileNetV2ForSemanticSegmentation', MobileNetV2ForSemanticSegmentation]],
    ['mobilenet_v3', ['MobileNetV3ForSemanticSegmentation', MobileNetV3ForSemanticSegmentation]],
    ['mobilenet_v4', ['MobileNetV4ForSemanticSegmentation', MobileNetV4ForSemanticSegmentation]],
]);

const MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES = new Map([
    ['detr', ['DetrForSegmentation', DetrForSegmentation]],
    ['maskformer', ['MaskFormerForInstanceSegmentation', MaskFormerForInstanceSegmentation]],
]);

const MODEL_FOR_MASK_GENERATION_MAPPING_NAMES = new Map([
    ['sam', ['SamModel', SamModel]],
    ['sam2', ['Sam2Model', Sam2Model]],
    ['edgetam', ['EdgeTamModel', EdgeTamModel]],
    ['sam3_tracker', ['Sam3TrackerModel', Sam3TrackerModel]],
]);

const MODEL_FOR_CTC_MAPPING_NAMES = new Map([
    ['wav2vec2', ['Wav2Vec2ForCTC', Wav2Vec2ForCTC]],
    ['wav2vec2-bert', ['Wav2Vec2BertForCTC', Wav2Vec2BertForCTC]],
    ['unispeech', ['UniSpeechForCTC', UniSpeechForCTC]],
    ['unispeech-sat', ['UniSpeechSatForCTC', UniSpeechSatForCTC]],
    ['wavlm', ['WavLMForCTC', WavLMForCTC]],
    ['hubert', ['HubertForCTC', HubertForCTC]],
    ['parakeet_ctc', ['ParakeetForCTC', ParakeetForCTC]],
]);

const MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES = new Map([
    ['wav2vec2', ['Wav2Vec2ForSequenceClassification', Wav2Vec2ForSequenceClassification]],
    ['wav2vec2-bert', ['Wav2Vec2BertForSequenceClassification', Wav2Vec2BertForSequenceClassification]],
    ['unispeech', ['UniSpeechForSequenceClassification', UniSpeechForSequenceClassification]],
    ['unispeech-sat', ['UniSpeechSatForSequenceClassification', UniSpeechSatForSequenceClassification]],
    ['wavlm', ['WavLMForSequenceClassification', WavLMForSequenceClassification]],
    ['hubert', ['HubertForSequenceClassification', HubertForSequenceClassification]],
    ['audio-spectrogram-transformer', ['ASTForAudioClassification', ASTForAudioClassification]],
]);

const MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES = new Map([['wavlm', ['WavLMForXVector', WavLMForXVector]]]);

const MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES = new Map([
    ['unispeech-sat', ['UniSpeechSatForAudioFrameClassification', UniSpeechSatForAudioFrameClassification]],
    ['wavlm', ['WavLMForAudioFrameClassification', WavLMForAudioFrameClassification]],
    ['wav2vec2', ['Wav2Vec2ForAudioFrameClassification', Wav2Vec2ForAudioFrameClassification]],
    ['pyannote', ['PyAnnoteForAudioFrameClassification', PyAnnoteForAudioFrameClassification]],
]);

const MODEL_FOR_IMAGE_MATTING_MAPPING_NAMES = new Map([
    ['vitmatte', ['VitMatteForImageMatting', VitMatteForImageMatting]],
]);

const MODEL_FOR_TIME_SERIES_PREDICTION_MAPPING_NAMES = new Map([
    ['patchtst', ['PatchTSTForPrediction', PatchTSTForPrediction]],
    ['patchtsmixer', ['PatchTSMixerForPrediction', PatchTSMixerForPrediction]],
]);

const MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES = new Map([
    ['swin2sr', ['Swin2SRForImageSuperResolution', Swin2SRForImageSuperResolution]],
]);

const MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES = new Map([
    ['dpt', ['DPTForDepthEstimation', DPTForDepthEstimation]],
    ['depth_anything', ['DepthAnythingForDepthEstimation', DepthAnythingForDepthEstimation]],
    ['glpn', ['GLPNForDepthEstimation', GLPNForDepthEstimation]],
    ['sapiens', ['SapiensForDepthEstimation', SapiensForDepthEstimation]],
    ['depth_pro', ['DepthProForDepthEstimation', DepthProForDepthEstimation]],
    ['metric3d', ['Metric3DForDepthEstimation', Metric3DForDepthEstimation]],
    ['metric3dv2', ['Metric3Dv2ForDepthEstimation', Metric3Dv2ForDepthEstimation]],
]);

const MODEL_FOR_NORMAL_ESTIMATION_MAPPING_NAMES = new Map([
    ['sapiens', ['SapiensForNormalEstimation', SapiensForNormalEstimation]],
]);

const MODEL_FOR_POSE_ESTIMATION_MAPPING_NAMES = new Map([
    ['vitpose', ['VitPoseForPoseEstimation', VitPoseForPoseEstimation]],
]);

// NOTE: This is custom to Transformers.js, and is necessary because certain models
// (e.g., CLIP) are split into vision and text components
const MODEL_FOR_IMAGE_FEATURE_EXTRACTION_MAPPING_NAMES = new Map([
    ['clip', ['CLIPVisionModelWithProjection', CLIPVisionModelWithProjection]],
    ['siglip', ['SiglipVisionModel', SiglipVisionModel]],
    ['jina_clip', ['JinaCLIPVisionModel', JinaCLIPVisionModel]],
]);

const MODEL_CLASS_TYPE_MAPPING = [
    // MODEL_MAPPING_NAMES:
    [MODEL_MAPPING_NAMES_ENCODER_ONLY, MODEL_TYPES.EncoderOnly],
    [MODEL_MAPPING_NAMES_ENCODER_DECODER, MODEL_TYPES.EncoderDecoder],
    [MODEL_MAPPING_NAMES_DECODER_ONLY, MODEL_TYPES.DecoderOnly],
    [MODEL_MAPPING_NAMES_AUTO_ENCODER, MODEL_TYPES.AutoEncoder],

    [MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
    [MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
    [MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES, MODEL_TYPES.Seq2Seq],
    [MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES, MODEL_TYPES.Seq2Seq],
    [MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_TYPES.DecoderOnly],
    [MODEL_FOR_MULTIMODALITY_MAPPING_NAMES, MODEL_TYPES.MultiModality],
    [MODEL_FOR_MASKED_LM_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
    [MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
    [MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES, MODEL_TYPES.Vision2Seq],
    [MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES, MODEL_TYPES.ImageTextToText],
    [MODEL_FOR_AUDIO_TEXT_TO_TEXT_MAPPING_NAMES, MODEL_TYPES.AudioTextToText],
    [MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
    [MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
    [MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
    [MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
    [MODEL_FOR_IMAGE_MATTING_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
    [MODEL_FOR_TIME_SERIES_PREDICTION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
    [MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
    [MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
    [MODEL_FOR_NORMAL_ESTIMATION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
    [MODEL_FOR_POSE_ESTIMATION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
    [MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
    [MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
    [MODEL_FOR_MASK_GENERATION_MAPPING_NAMES, MODEL_TYPES.MaskGeneration],
    [MODEL_FOR_CTC_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
    [MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
    [MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES, MODEL_TYPES.Seq2Seq],
    [MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
    [MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
    [MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],

    // Custom:
    [MODEL_FOR_IMAGE_FEATURE_EXTRACTION_MAPPING_NAMES, MODEL_TYPES.EncoderOnly],
];

for (const [mappings, type] of MODEL_CLASS_TYPE_MAPPING) {
    // @ts-ignore
    for (const [name, model] of mappings.values()) {
        MODEL_TYPE_MAPPING.set(name, type);
        MODEL_CLASS_TO_NAME_MAPPING.set(model, name);
        MODEL_NAME_TO_CLASS_MAPPING.set(name, model);
    }
}

const CUSTOM_MAPPING = [
    // OVERRIDE:
    // TODO: Refactor to allow class to specify model
    ['MusicgenForConditionalGeneration', MusicgenForConditionalGeneration, MODEL_TYPES.Musicgen],
    ['Phi3VForCausalLM', Phi3VForCausalLM, MODEL_TYPES.Phi3V],

    ['CLIPTextModelWithProjection', CLIPTextModelWithProjection, MODEL_TYPES.EncoderOnly],
    ['SiglipTextModel', SiglipTextModel, MODEL_TYPES.EncoderOnly],
    ['JinaCLIPTextModel', JinaCLIPTextModel, MODEL_TYPES.EncoderOnly],
    ['ClapTextModelWithProjection', ClapTextModelWithProjection, MODEL_TYPES.EncoderOnly],
    ['ClapAudioModelWithProjection', ClapAudioModelWithProjection, MODEL_TYPES.EncoderOnly],

    ['DacEncoderModel', DacEncoderModel, MODEL_TYPES.EncoderOnly],
    ['DacDecoderModel', DacDecoderModel, MODEL_TYPES.EncoderOnly],
    ['MimiEncoderModel', MimiEncoderModel, MODEL_TYPES.EncoderOnly],
    ['MimiDecoderModel', MimiDecoderModel, MODEL_TYPES.EncoderOnly],
    ['SnacEncoderModel', SnacEncoderModel, MODEL_TYPES.EncoderOnly],
    ['SnacDecoderModel', SnacDecoderModel, MODEL_TYPES.EncoderOnly],

    ['Gemma3nForConditionalGeneration', Gemma3nForConditionalGeneration, MODEL_TYPES.ImageAudioTextToText],
    ['SupertonicForConditionalGeneration', SupertonicForConditionalGeneration, MODEL_TYPES.Supertonic],
    ['ChatterboxModel', ChatterboxModel, MODEL_TYPES.Chatterbox],
];
for (const [name, model, type] of CUSTOM_MAPPING) {
    MODEL_TYPE_MAPPING.set(name, type);
    MODEL_CLASS_TO_NAME_MAPPING.set(model, name);
    MODEL_NAME_TO_CLASS_MAPPING.set(name, model);
}

const CUSTOM_ARCHITECTURES = new Map([
    ['modnet', MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES],
    ['birefnet', MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES],
    ['isnet', MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES],
    ['ben', MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES],
]);
for (const [name, mapping] of CUSTOM_ARCHITECTURES.entries()) {
    mapping.set(name, ['PreTrainedModel', PreTrainedModel]);
    MODEL_TYPE_MAPPING.set(name, MODEL_TYPES.EncoderOnly);
    MODEL_CLASS_TO_NAME_MAPPING.set(PreTrainedModel, name);
    MODEL_NAME_TO_CLASS_MAPPING.set(name, PreTrainedModel);
}

/**
 * Helper class which is used to instantiate pretrained models with the `from_pretrained` function.
 * The chosen model class is determined by the type specified in the model config.
 *
 * @example
 * let model = await AutoModel.from_pretrained('Xenova/bert-base-uncased');
 */
export class AutoModel extends PretrainedMixin {
    /** @type {Map<string, Object>[]} */
    // @ts-ignore
    static MODEL_CLASS_MAPPINGS = MODEL_CLASS_TYPE_MAPPING.map((x) => x[0]);
    static BASE_IF_FAIL = true;
}

/**
 * Helper class which is used to instantiate pretrained sequence classification models with the `from_pretrained` function.
 * The chosen model class is determined by the type specified in the model config.
 *
 * @example
 * let model = await AutoModelForSequenceClassification.from_pretrained('Xenova/distilbert-base-uncased-finetuned-sst-2-english');
 */
export class AutoModelForSequenceClassification extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES];
}

/**
 * Helper class which is used to instantiate pretrained token classification models with the `from_pretrained` function.
 * The chosen model class is determined by the type specified in the model config.
 *
 * @example
 * let model = await AutoModelForTokenClassification.from_pretrained('Xenova/distilbert-base-multilingual-cased-ner-hrl');
 */
export class AutoModelForTokenClassification extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES];
}

/**
 * Helper class which is used to instantiate pretrained sequence-to-sequence models with the `from_pretrained` function.
 * The chosen model class is determined by the type specified in the model config.
 *
 * @example
 * let model = await AutoModelForSeq2SeqLM.from_pretrained('Xenova/t5-small');
 */
export class AutoModelForSeq2SeqLM extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES];
}

/**
 * Helper class which is used to instantiate pretrained sequence-to-sequence speech-to-text models with the `from_pretrained` function.
 * The chosen model class is determined by the type specified in the model config.
 *
 * @example
 * let model = await AutoModelForSpeechSeq2Seq.from_pretrained('openai/whisper-tiny.en');
 */
export class AutoModelForSpeechSeq2Seq extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES];
}

/**
 * Helper class which is used to instantiate pretrained sequence-to-sequence text-to-spectrogram models with the `from_pretrained` function.
 * The chosen model class is determined by the type specified in the model config.
 *
 * @example
 * let model = await AutoModelForTextToSpectrogram.from_pretrained('microsoft/speecht5_tts');
 */
export class AutoModelForTextToSpectrogram extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES];
}

/**
 * Helper class which is used to instantiate pretrained text-to-waveform models with the `from_pretrained` function.
 * The chosen model class is determined by the type specified in the model config.
 *
 * @example
 * let model = await AutoModelForTextToSpectrogram.from_pretrained('facebook/mms-tts-eng');
 */
export class AutoModelForTextToWaveform extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES];
}

/**
 * Helper class which is used to instantiate pretrained causal language models with the `from_pretrained` function.
 * The chosen model class is determined by the type specified in the model config.
 *
 * @example
 * let model = await AutoModelForCausalLM.from_pretrained('Xenova/gpt2');
 */
export class AutoModelForCausalLM extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_FOR_CAUSAL_LM_MAPPING_NAMES];
}

/**
 * Helper class which is used to instantiate pretrained masked language models with the `from_pretrained` function.
 * The chosen model class is determined by the type specified in the model config.
 *
 * @example
 * let model = await AutoModelForMaskedLM.from_pretrained('Xenova/bert-base-uncased');
 */
export class AutoModelForMaskedLM extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_FOR_MASKED_LM_MAPPING_NAMES];
}

/**
 * Helper class which is used to instantiate pretrained question answering models with the `from_pretrained` function.
 * The chosen model class is determined by the type specified in the model config.
 *
 * @example
 * let model = await AutoModelForQuestionAnswering.from_pretrained('Xenova/distilbert-base-cased-distilled-squad');
 */
export class AutoModelForQuestionAnswering extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES];
}

/**
 * Helper class which is used to instantiate pretrained vision-to-sequence models with the `from_pretrained` function.
 * The chosen model class is determined by the type specified in the model config.
 *
 * @example
 * let model = await AutoModelForVision2Seq.from_pretrained('Xenova/vit-gpt2-image-captioning');
 */
export class AutoModelForVision2Seq extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES];
}

/**
 * Helper class which is used to instantiate pretrained image classification models with the `from_pretrained` function.
 * The chosen model class is determined by the type specified in the model config.
 *
 * @example
 * let model = await AutoModelForImageClassification.from_pretrained('Xenova/vit-base-patch16-224');
 */
export class AutoModelForImageClassification extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES];
}

/**
 * Helper class which is used to instantiate pretrained image segmentation models with the `from_pretrained` function.
 * The chosen model class is determined by the type specified in the model config.
 *
 * @example
 * let model = await AutoModelForImageSegmentation.from_pretrained('Xenova/detr-resnet-50-panoptic');
 */
export class AutoModelForImageSegmentation extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES];
}

/**
 * Helper class which is used to instantiate pretrained image segmentation models with the `from_pretrained` function.
 * The chosen model class is determined by the type specified in the model config.
 *
 * @example
 * let model = await AutoModelForSemanticSegmentation.from_pretrained('nvidia/segformer-b3-finetuned-cityscapes-1024-1024');
 */
export class AutoModelForSemanticSegmentation extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES];
}

/**
 * Helper class which is used to instantiate pretrained universal image segmentation models with the `from_pretrained` function.
 * The chosen model class is determined by the type specified in the model config.
 *
 * @example
 * let model = await AutoModelForUniversalSegmentation.from_pretrained('hf-internal-testing/tiny-random-MaskFormerForInstanceSegmentation');
 */
export class AutoModelForUniversalSegmentation extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES];
}

/**
 * Helper class which is used to instantiate pretrained object detection models with the `from_pretrained` function.
 * The chosen model class is determined by the type specified in the model config.
 *
 * @example
 * let model = await AutoModelForObjectDetection.from_pretrained('Xenova/detr-resnet-50');
 */
export class AutoModelForObjectDetection extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES];
}

export class AutoModelForZeroShotObjectDetection extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES];
}

/**
 * Helper class which is used to instantiate pretrained mask generation models with the `from_pretrained` function.
 * The chosen model class is determined by the type specified in the model config.
 *
 * @example
 * let model = await AutoModelForMaskGeneration.from_pretrained('Xenova/sam-vit-base');
 */
export class AutoModelForMaskGeneration extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_FOR_MASK_GENERATION_MAPPING_NAMES];
}

export class AutoModelForCTC extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_FOR_CTC_MAPPING_NAMES];
}

export class AutoModelForAudioClassification extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES];
}

export class AutoModelForXVector extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES];
}

export class AutoModelForAudioFrameClassification extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES];
}

export class AutoModelForDocumentQuestionAnswering extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES];
}

export class AutoModelForImageMatting extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_FOR_IMAGE_MATTING_MAPPING_NAMES];
}

export class AutoModelForImageToImage extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES];
}

export class AutoModelForDepthEstimation extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES];
}

export class AutoModelForNormalEstimation extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_FOR_NORMAL_ESTIMATION_MAPPING_NAMES];
}

export class AutoModelForPoseEstimation extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_FOR_POSE_ESTIMATION_MAPPING_NAMES];
}

export class AutoModelForImageFeatureExtraction extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_FOR_IMAGE_FEATURE_EXTRACTION_MAPPING_NAMES];
}

export class AutoModelForImageTextToText extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES];
}

export class AutoModelForAudioTextToText extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_FOR_AUDIO_TEXT_TO_TEXT_MAPPING_NAMES];
}

//////////////////////////////////////////////////

//////////////////////////////////////////////////
export class Seq2SeqLMOutput extends ModelOutput {
    /**
     * @param {Object} output The output of the model.
     * @param {Tensor} output.logits The output logits of the model.
     * @param {Tensor} output.past_key_values An tensor of key/value pairs that represent the previous state of the model.
     * @param {Tensor} output.encoder_outputs The output of the encoder in a sequence-to-sequence model.
     * @param {Tensor} [output.decoder_attentions] Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.
     * @param {Tensor} [output.cross_attentions] Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.
     */
    constructor({ logits, past_key_values, encoder_outputs, decoder_attentions = null, cross_attentions = null }) {
        super();
        this.logits = logits;
        this.past_key_values = past_key_values;
        this.encoder_outputs = encoder_outputs;
        this.decoder_attentions = decoder_attentions;
        this.cross_attentions = cross_attentions;
    }
}

export class ImageMattingOutput extends ModelOutput {
    /**
     * @param {Object} output The output of the model.
     * @param {Tensor} output.alphas Estimated alpha values, of shape `(batch_size, num_channels, height, width)`.
     */
    constructor({ alphas }) {
        super();
        this.alphas = alphas;
    }
}

/**
 * Describes the outputs for the VITS model.
 */
export class VitsModelOutput extends ModelOutput {
    /**
     * @param {Object} output The output of the model.
     * @param {Tensor} output.waveform The final audio waveform predicted by the model, of shape `(batch_size, sequence_length)`.
     * @param {Tensor} output.spectrogram The log-mel spectrogram predicted at the output of the flow model.
     * This spectrogram is passed to the Hi-Fi GAN decoder model to obtain the final audio waveform.
     */
    constructor({ waveform, spectrogram }) {
        super();
        this.waveform = waveform;
        this.spectrogram = spectrogram;
    }
}
