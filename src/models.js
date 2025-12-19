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

//////////////////////////////////////////////////

//////////////////////////////////////////////////
export class RFDetrPreTrainedModel extends PreTrainedModel {}
export class RFDetrModel extends RFDetrPreTrainedModel {}
export class RFDetrForObjectDetection extends RFDetrPreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new RFDetrObjectDetectionOutput(await super._call(model_inputs));
    }
}

export class RFDetrObjectDetectionOutput extends RTDetrObjectDetectionOutput {}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
export class DFinePreTrainedModel extends PreTrainedModel {}
export class DFineModel extends DFinePreTrainedModel {}
export class DFineForObjectDetection extends DFinePreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new RTDetrObjectDetectionOutput(await super._call(model_inputs));
    }
}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
export class TableTransformerPreTrainedModel extends PreTrainedModel {}

/**
 * The bare Table Transformer Model (consisting of a backbone and encoder-decoder Transformer)
 * outputting raw hidden-states without any specific head on top.
 */
export class TableTransformerModel extends TableTransformerPreTrainedModel {}

/**
 * Table Transformer Model (consisting of a backbone and encoder-decoder Transformer)
 * with object detection heads on top, for tasks such as COCO detection.
 */
export class TableTransformerForObjectDetection extends TableTransformerPreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new TableTransformerObjectDetectionOutput(await super._call(model_inputs));
    }
}
export class TableTransformerObjectDetectionOutput extends DetrObjectDetectionOutput {}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
export class DeiTPreTrainedModel extends PreTrainedModel {}
export class DeiTModel extends DeiTPreTrainedModel {}
export class DeiTForImageClassification extends DeiTPreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
export class HieraPreTrainedModel extends PreTrainedModel {}
export class HieraModel extends HieraPreTrainedModel {}
export class HieraForImageClassification extends HieraPreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
/**
 * An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained models.
 */
export class ResNetPreTrainedModel extends PreTrainedModel {}

/**
 * The bare ResNet model outputting raw features without any specific head on top.
 */
export class ResNetModel extends ResNetPreTrainedModel {}

/**
 * ResNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for ImageNet.
 */
export class ResNetForImageClassification extends ResNetPreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
export class SwinPreTrainedModel extends PreTrainedModel {}
export class SwinModel extends SwinPreTrainedModel {}
export class SwinForImageClassification extends SwinPreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
export class SwinForSemanticSegmentation extends SwinPreTrainedModel {}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
export class Swin2SRPreTrainedModel extends PreTrainedModel {}

/**
 * The bare Swin2SR Model transformer outputting raw hidden-states without any specific head on top.
 */
export class Swin2SRModel extends Swin2SRPreTrainedModel {}

/**
 * Swin2SR Model transformer with an upsampler head on top for image super resolution and restoration.
 *
 * **Example:** Super-resolution w/ `Xenova/swin2SR-classical-sr-x2-64`.
 *
 * ```javascript
 * import { AutoProcessor, Swin2SRForImageSuperResolution, RawImage } from '@huggingface/transformers';
 *
 * // Load processor and model
 * const model_id = 'Xenova/swin2SR-classical-sr-x2-64';
 * const processor = await AutoProcessor.from_pretrained(model_id);
 * const model = await Swin2SRForImageSuperResolution.from_pretrained(model_id);
 *
 * // Prepare model inputs
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/butterfly.jpg';
 * const image = await RawImage.fromURL(url);
 * const inputs = await processor(image);
 *
 * // Run model
 * const outputs = await model(inputs);
 *
 * // Convert Tensor to RawImage
 * const output = outputs.reconstruction.squeeze().clamp_(0, 1).mul_(255).round_().to('uint8');
 * const outputImage = RawImage.fromTensor(output);
 * // RawImage {
 * //   data: Uint8Array(786432) [ 41, 31, 24, ... ],
 * //   width: 512,
 * //   height: 512,
 * //   channels: 3
 * // }
 * ```
 */
export class Swin2SRForImageSuperResolution extends Swin2SRPreTrainedModel {}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
export class DPTPreTrainedModel extends PreTrainedModel {}

/**
 * The bare DPT Model transformer outputting raw hidden-states without any specific head on top.
 */
export class DPTModel extends DPTPreTrainedModel {}

/**
 * DPT Model with a depth estimation head on top (consisting of 3 convolutional layers) e.g. for KITTI, NYUv2.
 *
 * **Example:** Depth estimation w/ `Xenova/dpt-hybrid-midas`.
 * ```javascript
 * import { DPTForDepthEstimation, AutoProcessor, RawImage, interpolate_4d } from '@huggingface/transformers';
 *
 * // Load model and processor
 * const model_id = 'Xenova/dpt-hybrid-midas';
 * const model = await DPTForDepthEstimation.from_pretrained(model_id);
 * const processor = await AutoProcessor.from_pretrained(model_id);
 *
 * // Load image from URL
 * const url = 'http://images.cocodataset.org/val2017/000000039769.jpg';
 * const image = await RawImage.read(url);
 *
 * // Prepare image for the model
 * const inputs = await processor(image);
 *
 * // Run model
 * const { predicted_depth } = await model(inputs);
 *
 * // Interpolate to original size
 * const prediction = (await interpolate_4d(predicted_depth.unsqueeze(1), {
 * size: image.size.reverse(),
 * mode: 'bilinear',
 * })).squeeze(1);
 *
 * // Visualize the prediction
 * const min = prediction.min().item();
 * const max = prediction.max().item();
 * const formatted = prediction.sub_(min).div_(max - min).mul_(255).to('uint8');
 * const depth = RawImage.fromTensor(formatted);
 * // RawImage {
 * //   data: Uint8Array(307200) [ 85, 85, 84, ... ],
 * //   width: 640,
 * //   height: 480,
 * //   channels: 1
 * // }
 * ```
 */
export class DPTForDepthEstimation extends DPTPreTrainedModel {}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
export class DepthAnythingPreTrainedModel extends PreTrainedModel {}

/**
 * Depth Anything Model with a depth estimation head on top (consisting of 3 convolutional layers) e.g. for KITTI, NYUv2.
 */
export class DepthAnythingForDepthEstimation extends DepthAnythingPreTrainedModel {}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
export class SapiensPreTrainedModel extends PreTrainedModel {}
export class SapiensForSemanticSegmentation extends SapiensPreTrainedModel {}
export class SapiensForDepthEstimation extends SapiensPreTrainedModel {}
export class SapiensForNormalEstimation extends SapiensPreTrainedModel {}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
export class DepthProPreTrainedModel extends PreTrainedModel {}
export class DepthProForDepthEstimation extends DepthProPreTrainedModel {}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
export class Metric3DPreTrainedModel extends PreTrainedModel {}
export class Metric3DForDepthEstimation extends Metric3DPreTrainedModel {}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
export class Metric3Dv2PreTrainedModel extends PreTrainedModel {}
export class Metric3Dv2ForDepthEstimation extends Metric3Dv2PreTrainedModel {}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
export class MaskFormerPreTrainedModel extends PreTrainedModel {}
export class MaskFormerModel extends MaskFormerPreTrainedModel {}
export class MaskFormerForInstanceSegmentation extends MaskFormerPreTrainedModel {}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
export class GLPNPreTrainedModel extends PreTrainedModel {}

/**
 * The bare GLPN encoder (Mix-Transformer) outputting raw hidden-states without any specific head on top.
 */
export class GLPNModel extends GLPNPreTrainedModel {}

/**
 * import { GLPNForDepthEstimation, AutoProcessor, RawImage, interpolate_4d } from '@huggingface/transformers';
 *
 * // Load model and processor
 * const model_id = 'Xenova/glpn-kitti';
 * const model = await GLPNForDepthEstimation.from_pretrained(model_id);
 * const processor = await AutoProcessor.from_pretrained(model_id);
 *
 * // Load image from URL
 * const url = 'http://images.cocodataset.org/val2017/000000039769.jpg';
 * const image = await RawImage.read(url);
 *
 * // Prepare image for the model
 * const inputs = await processor(image);
 *
 * // Run model
 * const { predicted_depth } = await model(inputs);
 *
 * // Interpolate to original size
 * const prediction = (await interpolate_4d(predicted_depth.unsqueeze(1), {
 * size: image.size.reverse(),
 * mode: 'bilinear',
 * })).squeeze(1);
 *
 * // Visualize the prediction
 * const min = prediction.min().item();
 * const max = prediction.max().item();
 * const formatted = prediction.sub_(min).div_(max - min).mul_(255).to('uint8');
 * const depth = RawImage.fromTensor(formatted);
 * // RawImage {
 * //   data: Uint8Array(307200) [ 85, 85, 84, ... ],
 * //   width: 640,
 * //   height: 480,
 * //   channels: 1
 * // }
 * ```
 */
export class GLPNForDepthEstimation extends GLPNPreTrainedModel {}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
export class DonutSwinPreTrainedModel extends PreTrainedModel {}

/**
 * The bare Donut Swin Model transformer outputting raw hidden-states without any specific head on top.
 *
 * **Example:** Step-by-step Document Parsing.
 *
 * ```javascript
 * import { AutoProcessor, AutoTokenizer, AutoModelForVision2Seq, RawImage } from '@huggingface/transformers';
 *
 * // Choose model to use
 * const model_id = 'Xenova/donut-base-finetuned-cord-v2';
 *
 * // Prepare image inputs
 * const processor = await AutoProcessor.from_pretrained(model_id);
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/receipt.png';
 * const image = await RawImage.read(url);
 * const image_inputs = await processor(image);
 *
 * // Prepare decoder inputs
 * const tokenizer = await AutoTokenizer.from_pretrained(model_id);
 * const task_prompt = '<s_cord-v2>';
 * const decoder_input_ids = tokenizer(task_prompt, {
 *   add_special_tokens: false,
 * }).input_ids;
 *
 * // Create the model
 * const model = await AutoModelForVision2Seq.from_pretrained(model_id);
 *
 * // Run inference
 * const output = await model.generate(image_inputs.pixel_values, {
 *   decoder_input_ids,
 *   max_length: model.config.decoder.max_position_embeddings,
 * });
 *
 * // Decode output
 * const decoded = tokenizer.batch_decode(output)[0];
 * // <s_cord-v2><s_menu><s_nm> CINNAMON SUGAR</s_nm><s_unitprice> 17,000</s_unitprice><s_cnt> 1 x</s_cnt><s_price> 17,000</s_price></s_menu><s_sub_total><s_subtotal_price> 17,000</s_subtotal_price></s_sub_total><s_total><s_total_price> 17,000</s_total_price><s_cashprice> 20,000</s_cashprice><s_changeprice> 3,000</s_changeprice></s_total></s>
 * ```
 *
 * **Example:** Step-by-step Document Visual Question Answering (DocVQA)
 *
 * ```javascript
 * import { AutoProcessor, AutoTokenizer, AutoModelForVision2Seq, RawImage } from '@huggingface/transformers';
 *
 * // Choose model to use
 * const model_id = 'Xenova/donut-base-finetuned-docvqa';
 *
 * // Prepare image inputs
 * const processor = await AutoProcessor.from_pretrained(model_id);
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/invoice.png';
 * const image = await RawImage.read(url);
 * const image_inputs = await processor(image);
 *
 * // Prepare decoder inputs
 * const tokenizer = await AutoTokenizer.from_pretrained(model_id);
 * const question = 'What is the invoice number?';
 * const task_prompt = `<s_docvqa><s_question>${question}</s_question><s_answer>`;
 * const decoder_input_ids = tokenizer(task_prompt, {
 *   add_special_tokens: false,
 * }).input_ids;
 *
 * // Create the model
 * const model = await AutoModelForVision2Seq.from_pretrained(model_id);
 *
 * // Run inference
 * const output = await model.generate(image_inputs.pixel_values, {
 *   decoder_input_ids,
 *   max_length: model.config.decoder.max_position_embeddings,
 * });
 *
 * // Decode output
 * const decoded = tokenizer.batch_decode(output)[0];
 * // <s_docvqa><s_question> What is the invoice number?</s_question><s_answer> us-001</s_answer></s>
 * ```
 */
export class DonutSwinModel extends DonutSwinPreTrainedModel {}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
export class ConvNextPreTrainedModel extends PreTrainedModel {}

/**
 * The bare ConvNext model outputting raw features without any specific head on top.
 */
export class ConvNextModel extends ConvNextPreTrainedModel {}

/**
 * ConvNext Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for ImageNet.
 */
export class ConvNextForImageClassification extends ConvNextPreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
export class ConvNextV2PreTrainedModel extends PreTrainedModel {}

/**
 * The bare ConvNextV2 model outputting raw features without any specific head on top.
 */
export class ConvNextV2Model extends ConvNextV2PreTrainedModel {}

/**
 * ConvNextV2 Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for ImageNet.
 */
export class ConvNextV2ForImageClassification extends ConvNextV2PreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
export class Dinov2PreTrainedModel extends PreTrainedModel {}

/**
 * The bare DINOv2 Model transformer outputting raw hidden-states without any specific head on top.
 */
export class Dinov2Model extends Dinov2PreTrainedModel {}

/**
 * Dinov2 Model transformer with an image classification head on top (a linear layer on top of the final hidden state of the [CLS] token) e.g. for ImageNet.
 */
export class Dinov2ForImageClassification extends Dinov2PreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
export class Dinov2WithRegistersPreTrainedModel extends PreTrainedModel {}

/**
 * The bare Dinov2WithRegisters Model transformer outputting raw hidden-states without any specific head on top.
 */
export class Dinov2WithRegistersModel extends Dinov2WithRegistersPreTrainedModel {}

/**
 * Dinov2WithRegisters Model transformer with an image classification head on top (a linear layer on top of the final hidden state of the [CLS] token) e.g. for ImageNet.
 */
export class Dinov2WithRegistersForImageClassification extends Dinov2WithRegistersPreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
export class DINOv3ViTPreTrainedModel extends PreTrainedModel {}
export class DINOv3ViTModel extends DINOv3ViTPreTrainedModel {}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
export class DINOv3ConvNextPreTrainedModel extends PreTrainedModel {}
export class DINOv3ConvNextModel extends DINOv3ConvNextPreTrainedModel {}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
export class GroundingDinoPreTrainedModel extends PreTrainedModel {}
export class GroundingDinoForObjectDetection extends GroundingDinoPreTrainedModel {}

//////////////////////////////////////////////////
export class YolosPreTrainedModel extends PreTrainedModel {}
export class YolosModel extends YolosPreTrainedModel {}
export class YolosForObjectDetection extends YolosPreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new YolosObjectDetectionOutput(await super._call(model_inputs));
    }
}

export class YolosObjectDetectionOutput extends ModelOutput {
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
export class SamPreTrainedModel extends PreTrainedModel {}

/**
 * Segment Anything Model (SAM) for generating segmentation masks, given an input image
 * and optional 2D location and bounding boxes.
 *
 * **Example:** Perform mask generation w/ `Xenova/sam-vit-base`.
 * ```javascript
 * import { SamModel, AutoProcessor, RawImage } from '@huggingface/transformers';
 *
 * const model = await SamModel.from_pretrained('Xenova/sam-vit-base');
 * const processor = await AutoProcessor.from_pretrained('Xenova/sam-vit-base');
 *
 * const img_url = 'https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png';
 * const raw_image = await RawImage.read(img_url);
 * const input_points = [[[450, 600]]] // 2D localization of a window
 *
 * const inputs = await processor(raw_image, { input_points });
 * const outputs = await model(inputs);
 *
 * const masks = await processor.post_process_masks(outputs.pred_masks, inputs.original_sizes, inputs.reshaped_input_sizes);
 * // [
 * //   Tensor {
 * //     dims: [ 1, 3, 1764, 2646 ],
 * //     type: 'bool',
 * //     data: Uint8Array(14002632) [ ... ],
 * //     size: 14002632
 * //   }
 * // ]
 * const scores = outputs.iou_scores;
 * // Tensor {
 * //   dims: [ 1, 1, 3 ],
 * //   type: 'float32',
 * //   data: Float32Array(3) [
 * //     0.8892380595207214,
 * //     0.9311248064041138,
 * //     0.983696699142456
 * //   ],
 * //   size: 3
 * // }
 * ```
 */
export class SamModel extends SamPreTrainedModel {
    /**
     * Compute image embeddings and positional image embeddings, given the pixel values of an image.
     * @param {Object} model_inputs Object containing the model inputs.
     * @param {Tensor} model_inputs.pixel_values Pixel values obtained using a `SamProcessor`.
     * @returns {Promise<{ image_embeddings: Tensor, image_positional_embeddings: Tensor }>} The image embeddings and positional image embeddings.
     */
    async get_image_embeddings({ pixel_values }) {
        // in:
        //  - pixel_values: tensor.float32[batch_size,3,1024,1024]
        //
        // out:
        //  - image_embeddings: tensor.float32[batch_size,256,64,64]
        //  - image_positional_embeddings: tensor.float32[batch_size,256,64,64]
        return await encoderForward(this, { pixel_values });
    }

    /**
     * @typedef {Object} SamModelInputs Object containing the model inputs.
     * @property {Tensor} pixel_values Pixel values as a Tensor with shape `(batch_size, num_channels, height, width)`.
     * These can be obtained using a `SamProcessor`.
     * @property {Tensor} [input_points] Input 2D spatial points with shape `(batch_size, num_points, 2)`.
     * This is used by the prompt encoder to encode the prompt.
     * @property {Tensor} [input_labels] Input labels for the points, as a Tensor of shape `(batch_size, point_batch_size, num_points)`.
     * This is used by the prompt encoder to encode the prompt. There are 4 types of labels:
     *  - `1`: the point is a point that contains the object of interest
     *  - `0`: the point is a point that does not contain the object of interest
     *  - `-1`: the point corresponds to the background
     *  - `-10`: the point is a padding point, thus should be ignored by the prompt encoder
     * @property {Tensor} [input_boxes] Input bounding boxes with shape `(batch_size, num_boxes, 4)`.
     * @property {Tensor} [image_embeddings] Image embeddings used by the mask decoder.
     * @property {Tensor} [image_positional_embeddings] Image positional embeddings used by the mask decoder.
     */

    /**
     * @param {SamModelInputs} model_inputs Object containing the model inputs.
     * @returns {Promise<Object>} The output of the model.
     */
    async forward(model_inputs) {
        if (!model_inputs.image_embeddings || !model_inputs.image_positional_embeddings) {
            // Compute the image embeddings if they are missing
            model_inputs = {
                ...model_inputs,
                ...(await this.get_image_embeddings(model_inputs)),
            };
        } else {
            model_inputs = { ...model_inputs };
        }

        // Set default input labels if they are missing
        model_inputs.input_labels ??= ones(model_inputs.input_points.dims.slice(0, -1));

        const decoder_inputs = {
            image_embeddings: model_inputs.image_embeddings,
            image_positional_embeddings: model_inputs.image_positional_embeddings,
        };
        if (model_inputs.input_points) {
            decoder_inputs.input_points = model_inputs.input_points;
        }
        if (model_inputs.input_labels) {
            decoder_inputs.input_labels = model_inputs.input_labels;
        }
        if (model_inputs.input_boxes) {
            decoder_inputs.input_boxes = model_inputs.input_boxes;
        }

        // Returns:
        //  - iou_scores: tensor.float32[batch_size,point_batch_size,3]
        //  - pred_masks: tensor.float32[batch_size,point_batch_size,3,256,256]
        return await sessionRun(this.sessions['prompt_encoder_mask_decoder'], decoder_inputs);
    }

    /**
     * Runs the model with the provided inputs
     * @param {Object} model_inputs Model inputs
     * @returns {Promise<SamImageSegmentationOutput>} Object containing segmentation outputs
     */
    async _call(model_inputs) {
        return new SamImageSegmentationOutput(await super._call(model_inputs));
    }
}

//////////////////////////////////////////////////

//////////////////////////////////////////////////
export class Sam2ImageSegmentationOutput extends ModelOutput {
    /**
     * @param {Object} output The output of the model.
     * @param {Tensor} output.iou_scores The output logits of the model.
     * @param {Tensor} output.pred_masks Predicted boxes.
     * @param {Tensor} output.object_score_logits Logits for the object score, indicating if an object is present.
     */
    constructor({ iou_scores, pred_masks, object_score_logits }) {
        super();
        this.iou_scores = iou_scores;
        this.pred_masks = pred_masks;
        this.object_score_logits = object_score_logits;
    }
}

export class Sam2PreTrainedModel extends PreTrainedModel {}
export class Sam2Model extends Sam2PreTrainedModel {
    /**
     * Compute image embeddings and positional image embeddings, given the pixel values of an image.
     * @param {Object} model_inputs Object containing the model inputs.
     * @param {Tensor} model_inputs.pixel_values Pixel values obtained using a `Sam2Processor`.
     * @returns {Promise<Record<String, Tensor>>} The image embeddings.
     */
    async get_image_embeddings({ pixel_values }) {
        // in:
        //  - pixel_values: tensor.float32[batch_size,3,1024,1024]
        //
        // out:
        //  - image_embeddings.0: tensor.float32[batch_size,32,256,256]
        //  - image_embeddings.1: tensor.float32[batch_size,64,128,128]
        //  - image_embeddings.2: tensor.float32[batch_size,256,64,64]
        return await encoderForward(this, { pixel_values });
    }

    async forward(model_inputs) {
        // @ts-expect-error ts(2339)
        const { num_feature_levels } = this.config.vision_config;
        const image_embeddings_name = Array.from({ length: num_feature_levels }, (_, i) => `image_embeddings.${i}`);

        if (image_embeddings_name.some((name) => !model_inputs[name])) {
            // Compute the image embeddings if they are missing
            model_inputs = {
                ...model_inputs,
                ...(await this.get_image_embeddings(model_inputs)),
            };
        } else {
            model_inputs = { ...model_inputs };
        }

        if (model_inputs.input_points) {
            if (model_inputs.input_boxes && model_inputs.input_boxes.dims[1] !== 1) {
                throw new Error(
                    'When both `input_points` and `input_boxes` are provided, the number of boxes per image must be 1.',
                );
            }
            const shape = model_inputs.input_points.dims;
            model_inputs.input_labels ??= ones(shape.slice(0, -1));
            model_inputs.input_boxes ??= full([shape[0], 0, 4], 0.0);
        } else if (model_inputs.input_boxes) {
            // only boxes
            const shape = model_inputs.input_boxes.dims;
            model_inputs.input_labels = full([shape[0], shape[1], 0], -1n);
            model_inputs.input_points = full([shape[0], 1, 0, 2], 0.0);
        } else {
            throw new Error('At least one of `input_points` or `input_boxes` must be provided.');
        }

        const prompt_encoder_mask_decoder_session = this.sessions['prompt_encoder_mask_decoder'];
        const decoder_inputs = pick(model_inputs, prompt_encoder_mask_decoder_session.inputNames);

        // Returns:
        //  - iou_scores: tensor.float32[batch_size,num_boxes_or_points,3]
        //  - pred_masks: tensor.float32[batch_size,num_boxes_or_points,3,256,256]
        //  - object_score_logits: tensor.float32[batch_size,num_boxes_or_points,1]
        return await sessionRun(prompt_encoder_mask_decoder_session, decoder_inputs);
    }

    /**
     * Runs the model with the provided inputs
     * @param {Object} model_inputs Model inputs
     * @returns {Promise<Sam2ImageSegmentationOutput>} Object containing segmentation outputs
     */
    async _call(model_inputs) {
        return new Sam2ImageSegmentationOutput(await super._call(model_inputs));
    }
}
export class EdgeTamModel extends Sam2Model {} // NOTE: extends Sam2Model
export class Sam3TrackerModel extends Sam2Model {} // NOTE: extends Sam2Model
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// MarianMT models
export class MarianPreTrainedModel extends PreTrainedModel {}

export class MarianModel extends MarianPreTrainedModel {}

export class MarianMTModel extends MarianPreTrainedModel {}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// M2M100 models
export class M2M100PreTrainedModel extends PreTrainedModel {}

export class M2M100Model extends M2M100PreTrainedModel {}

export class M2M100ForConditionalGeneration extends M2M100PreTrainedModel {}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// Wav2Vec2 models
export class Wav2Vec2PreTrainedModel extends PreTrainedModel {}

/**
 * The bare Wav2Vec2 Model transformer outputting raw hidden-states without any specific head on top.
 *
 * **Example:** Load and run a `Wav2Vec2Model` for feature extraction.
 *
 * ```javascript
 * import { AutoProcessor, AutoModel, read_audio } from '@huggingface/transformers';
 *
 * // Read and preprocess audio
 * const processor = await AutoProcessor.from_pretrained('Xenova/mms-300m');
 * const audio = await read_audio('https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac', 16000);
 * const inputs = await processor(audio);
 *
 * // Run model with inputs
 * const model = await AutoModel.from_pretrained('Xenova/mms-300m');
 * const output = await model(inputs);
 * // {
 * //   last_hidden_state: Tensor {
 * //     dims: [ 1, 1144, 1024 ],
 * //     type: 'float32',
 * //     data: Float32Array(1171456) [ ... ],
 * //     size: 1171456
 * //   }
 * // }
 * ```
 */
export class Wav2Vec2Model extends Wav2Vec2PreTrainedModel {}

export class Wav2Vec2ForCTC extends Wav2Vec2PreTrainedModel {
    /**
     * @param {Object} model_inputs
     * @param {Tensor} model_inputs.input_values Float values of input raw speech waveform.
     * @param {Tensor} model_inputs.attention_mask Mask to avoid performing convolution and attention on padding token indices. Mask values selected in [0, 1]
     */
    async _call(model_inputs) {
        return new CausalLMOutput(await super._call(model_inputs));
    }
}

export class Wav2Vec2ForSequenceClassification extends Wav2Vec2PreTrainedModel {
    /**
     * Calls the model on new inputs.
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<SequenceClassifierOutput>} An object containing the model's output logits for sequence classification.
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}

/**
 * Wav2Vec2 Model with a frame classification head on top for tasks like Speaker Diarization.
 */
export class Wav2Vec2ForAudioFrameClassification extends Wav2Vec2PreTrainedModel {
    /**
     * Calls the model on new inputs.
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<TokenClassifierOutput>} An object containing the model's output logits for sequence classification.
     */
    async _call(model_inputs) {
        return new TokenClassifierOutput(await super._call(model_inputs));
    }
}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// Parakeet models
export class ParakeetPreTrainedModel extends PreTrainedModel {}
export class ParakeetForCTC extends ParakeetPreTrainedModel {
    /**
     * @param {Object} model_inputs
     * @param {Tensor} model_inputs.input_values Float values of input raw speech waveform.
     * @param {Tensor} model_inputs.attention_mask Mask to avoid performing convolution and attention on padding token indices. Mask values selected in [0, 1]
     */
    async _call(model_inputs) {
        return new CausalLMOutput(await super._call(model_inputs));
    }
}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// PyAnnote models
export class PyAnnotePreTrainedModel extends PreTrainedModel {}

/**
 * The bare PyAnnote Model transformer outputting raw hidden-states without any specific head on top.
 */
export class PyAnnoteModel extends PyAnnotePreTrainedModel {}

/**
 * PyAnnote Model with a frame classification head on top for tasks like Speaker Diarization.
 *
 * **Example:** Load and run a `PyAnnoteForAudioFrameClassification` for speaker diarization.
 *
 * ```javascript
 * import { AutoProcessor, AutoModelForAudioFrameClassification, read_audio } from '@huggingface/transformers';
 *
 * // Load model and processor
 * const model_id = 'onnx-community/pyannote-segmentation-3.0';
 * const model = await AutoModelForAudioFrameClassification.from_pretrained(model_id);
 * const processor = await AutoProcessor.from_pretrained(model_id);
 *
 * // Read and preprocess audio
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/mlk.wav';
 * const audio = await read_audio(url, processor.feature_extractor.config.sampling_rate);
 * const inputs = await processor(audio);
 *
 * // Run model with inputs
 * const { logits } = await model(inputs);
 * // {
 * //   logits: Tensor {
 * //     dims: [ 1, 767, 7 ],  // [batch_size, num_frames, num_classes]
 * //     type: 'float32',
 * //     data: Float32Array(5369) [ ... ],
 * //     size: 5369
 * //   }
 * // }
 *
 * const result = processor.post_process_speaker_diarization(logits, audio.length);
 * // [
 * //   [
 * //     { id: 0, start: 0, end: 1.0512535626298245, confidence: 0.8220156481664611 },
 * //     { id: 2, start: 1.0512535626298245, end: 2.3398869619825127, confidence: 0.9008811707860472 },
 * //     ...
 * //   ]
 * // ]
 *
 * // Display result
 * console.table(result[0], ['start', 'end', 'id', 'confidence']);
 * // ┌─────────┬────────────────────┬────────────────────┬────┬─────────────────────┐
 * // │ (index) │ start              │ end                │ id │ confidence          │
 * // ├─────────┼────────────────────┼────────────────────┼────┼─────────────────────┤
 * // │ 0       │ 0                  │ 1.0512535626298245 │ 0  │ 0.8220156481664611  │
 * // │ 1       │ 1.0512535626298245 │ 2.3398869619825127 │ 2  │ 0.9008811707860472  │
 * // │ 2       │ 2.3398869619825127 │ 3.5946089560890773 │ 0  │ 0.7521651315796233  │
 * // │ 3       │ 3.5946089560890773 │ 4.578039708226655  │ 2  │ 0.8491978128022479  │
 * // │ 4       │ 4.578039708226655  │ 4.594995410849717  │ 0  │ 0.2935352600416393  │
 * // │ 5       │ 4.594995410849717  │ 6.121008646925269  │ 3  │ 0.6788051309866024  │
 * // │ 6       │ 6.121008646925269  │ 6.256654267909762  │ 0  │ 0.37125512393851134 │
 * // │ 7       │ 6.256654267909762  │ 8.630452635138397  │ 2  │ 0.7467035186353542  │
 * // │ 8       │ 8.630452635138397  │ 10.088643060721703 │ 0  │ 0.7689364814666032  │
 * // │ 9       │ 10.088643060721703 │ 12.58113134631177  │ 2  │ 0.9123324509131324  │
 * // │ 10      │ 12.58113134631177  │ 13.005023911888312 │ 0  │ 0.4828358177572041  │
 * // └─────────┴────────────────────┴────────────────────┴────┴─────────────────────┘
 * ```
 */
export class PyAnnoteForAudioFrameClassification extends PyAnnotePreTrainedModel {
    /**
     * Calls the model on new inputs.
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<TokenClassifierOutput>} An object containing the model's output logits for sequence classification.
     */
    async _call(model_inputs) {
        return new TokenClassifierOutput(await super._call(model_inputs));
    }
}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// WeSpeakerResNet models
export class WeSpeakerResNetPreTrainedModel extends PreTrainedModel {}
export class WeSpeakerResNetModel extends WeSpeakerResNetPreTrainedModel {}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// UniSpeech models
export class UniSpeechPreTrainedModel extends PreTrainedModel {}

/**
 * The bare UniSpeech Model transformer outputting raw hidden-states without any specific head on top.
 */
export class UniSpeechModel extends UniSpeechPreTrainedModel {}

/**
 * UniSpeech Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).
 */
export class UniSpeechForCTC extends UniSpeechPreTrainedModel {
    /**
     * @param {Object} model_inputs
     * @param {Tensor} model_inputs.input_values Float values of input raw speech waveform.
     * @param {Tensor} model_inputs.attention_mask Mask to avoid performing convolution and attention on padding token indices. Mask values selected in [0, 1]
     */
    async _call(model_inputs) {
        return new CausalLMOutput(await super._call(model_inputs));
    }
}

/**
 * UniSpeech Model with a sequence classification head on top (a linear layer over the pooled output).
 */
export class UniSpeechForSequenceClassification extends UniSpeechPreTrainedModel {
    /**
     * Calls the model on new inputs.
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<SequenceClassifierOutput>} An object containing the model's output logits for sequence classification.
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// UniSpeechSat models
export class UniSpeechSatPreTrainedModel extends PreTrainedModel {}

/**
 * The bare UniSpeechSat Model transformer outputting raw hidden-states without any specific head on top.
 */
export class UniSpeechSatModel extends UniSpeechSatPreTrainedModel {}

/**
 * UniSpeechSat Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).
 */
export class UniSpeechSatForCTC extends UniSpeechSatPreTrainedModel {
    /**
     * @param {Object} model_inputs
     * @param {Tensor} model_inputs.input_values Float values of input raw speech waveform.
     * @param {Tensor} model_inputs.attention_mask Mask to avoid performing convolution and attention on padding token indices. Mask values selected in [0, 1]
     */
    async _call(model_inputs) {
        return new CausalLMOutput(await super._call(model_inputs));
    }
}

/**
 * UniSpeechSat Model with a sequence classification head on top (a linear layer over the pooled output).
 */
export class UniSpeechSatForSequenceClassification extends UniSpeechSatPreTrainedModel {
    /**
     * Calls the model on new inputs.
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<SequenceClassifierOutput>} An object containing the model's output logits for sequence classification.
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}

/**
 * UniSpeechSat Model with a frame classification head on top for tasks like Speaker Diarization.
 */
export class UniSpeechSatForAudioFrameClassification extends UniSpeechSatPreTrainedModel {
    /**
     * Calls the model on new inputs.
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<TokenClassifierOutput>} An object containing the model's output logits for sequence classification.
     */
    async _call(model_inputs) {
        return new TokenClassifierOutput(await super._call(model_inputs));
    }
}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// Wav2Vec2Bert models
export class Wav2Vec2BertPreTrainedModel extends PreTrainedModel {}

/**
 * The bare Wav2Vec2Bert Model transformer outputting raw hidden-states without any specific head on top.
 */
export class Wav2Vec2BertModel extends Wav2Vec2BertPreTrainedModel {}

/**
 * Wav2Vec2Bert Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).
 */
export class Wav2Vec2BertForCTC extends Wav2Vec2BertPreTrainedModel {
    /**
     * @param {Object} model_inputs
     * @param {Tensor} model_inputs.input_features Float values of input mel-spectrogram.
     * @param {Tensor} model_inputs.attention_mask Mask to avoid performing convolution and attention on padding token indices. Mask values selected in [0, 1]
     */
    async _call(model_inputs) {
        return new CausalLMOutput(await super._call(model_inputs));
    }
}

/**
 * Wav2Vec2Bert Model with a sequence classification head on top (a linear layer over the pooled output).
 */
export class Wav2Vec2BertForSequenceClassification extends Wav2Vec2BertPreTrainedModel {
    /**
     * Calls the model on new inputs.
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<SequenceClassifierOutput>} An object containing the model's output logits for sequence classification.
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// Hubert models
export class HubertPreTrainedModel extends PreTrainedModel {}

/**
 * The bare Hubert Model transformer outputting raw hidden-states without any specific head on top.
 *
 * **Example:** Load and run a `HubertModel` for feature extraction.
 *
 * ```javascript
 * import { AutoProcessor, AutoModel, read_audio } from '@huggingface/transformers';
 *
 * // Read and preprocess audio
 * const processor = await AutoProcessor.from_pretrained('Xenova/hubert-base-ls960');
 * const audio = await read_audio('https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav', 16000);
 * const inputs = await processor(audio);
 *
 * // Load and run model with inputs
 * const model = await AutoModel.from_pretrained('Xenova/hubert-base-ls960');
 * const output = await model(inputs);
 * // {
 * //   last_hidden_state: Tensor {
 * //     dims: [ 1, 549, 768 ],
 * //     type: 'float32',
 * //     data: Float32Array(421632) [0.0682469978928566, 0.08104046434164047, -0.4975186586380005, ...],
 * //     size: 421632
 * //   }
 * // }
 * ```
 */
export class HubertModel extends Wav2Vec2PreTrainedModel {}

/**
 * Hubert Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).
 */
export class HubertForCTC extends Wav2Vec2PreTrainedModel {
    /**
     * @param {Object} model_inputs
     * @param {Tensor} model_inputs.input_values Float values of input raw speech waveform.
     * @param {Tensor} model_inputs.attention_mask Mask to avoid performing convolution and attention on padding token indices. Mask values selected in [0, 1]
     */
    async _call(model_inputs) {
        return new CausalLMOutput(await super._call(model_inputs));
    }
}

/**
 * Hubert Model with a sequence classification head on top (a linear layer over the pooled output) for tasks like SUPERB Keyword Spotting.
 */
export class HubertForSequenceClassification extends Wav2Vec2PreTrainedModel {
    /**
     * Calls the model on new inputs.
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<SequenceClassifierOutput>} An object containing the model's output logits for sequence classification.
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// WavLM models
/**
 * An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained models.
 */
export class WavLMPreTrainedModel extends PreTrainedModel {}

/**
 * The bare WavLM Model transformer outputting raw hidden-states without any specific head on top.
 *
 * **Example:** Load and run a `WavLMModel` for feature extraction.
 *
 * ```javascript
 * import { AutoProcessor, AutoModel, read_audio } from '@huggingface/transformers';
 *
 * // Read and preprocess audio
 * const processor = await AutoProcessor.from_pretrained('Xenova/wavlm-base');
 * const audio = await read_audio('https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav', 16000);
 * const inputs = await processor(audio);
 *
 * // Run model with inputs
 * const model = await AutoModel.from_pretrained('Xenova/wavlm-base');
 * const output = await model(inputs);
 * // {
 * //   last_hidden_state: Tensor {
 * //     dims: [ 1, 549, 768 ],
 * //     type: 'float32',
 * //     data: Float32Array(421632) [-0.349443256855011, -0.39341306686401367,  0.022836603224277496, ...],
 * //     size: 421632
 * //   }
 * // }
 * ```
 */
export class WavLMModel extends WavLMPreTrainedModel {}

/**
 * WavLM Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).
 */
export class WavLMForCTC extends WavLMPreTrainedModel {
    /**
     * @param {Object} model_inputs
     * @param {Tensor} model_inputs.input_values Float values of input raw speech waveform.
     * @param {Tensor} model_inputs.attention_mask Mask to avoid performing convolution and attention on padding token indices. Mask values selected in [0, 1]
     */
    async _call(model_inputs) {
        return new CausalLMOutput(await super._call(model_inputs));
    }
}

/**
 * WavLM Model with a sequence classification head on top (a linear layer over the pooled output).
 */
export class WavLMForSequenceClassification extends WavLMPreTrainedModel {
    /**
     * Calls the model on new inputs.
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<SequenceClassifierOutput>} An object containing the model's output logits for sequence classification.
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}

/**
 * WavLM Model with an XVector feature extraction head on top for tasks like Speaker Verification.
 *
 * **Example:** Extract speaker embeddings with `WavLMForXVector`.
 * ```javascript
 * import { AutoProcessor, AutoModel, read_audio } from '@huggingface/transformers';
 *
 * // Read and preprocess audio
 * const processor = await AutoProcessor.from_pretrained('Xenova/wavlm-base-plus-sv');
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav';
 * const audio = await read_audio(url, 16000);
 * const inputs = await processor(audio);
 *
 * // Run model with inputs
 * const model = await AutoModel.from_pretrained('Xenova/wavlm-base-plus-sv');
 * const outputs = await model(inputs);
 * // {
 * //   logits: Tensor {
 * //     dims: [ 1, 512 ],
 * //     type: 'float32',
 * //     data: Float32Array(512) [0.5847219228744507, ...],
 * //     size: 512
 * //   },
 * //   embeddings: Tensor {
 * //     dims: [ 1, 512 ],
 * //     type: 'float32',
 * //     data: Float32Array(512) [-0.09079201519489288, ...],
 * //     size: 512
 * //   }
 * // }
 * ```
 */
export class WavLMForXVector extends WavLMPreTrainedModel {
    /**
     * Calls the model on new inputs.
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<XVectorOutput>} An object containing the model's output logits and speaker embeddings.
     */
    async _call(model_inputs) {
        return new XVectorOutput(await super._call(model_inputs));
    }
}

/**
 * WavLM Model with a frame classification head on top for tasks like Speaker Diarization.
 *
 * **Example:** Perform speaker diarization with `WavLMForAudioFrameClassification`.
 * ```javascript
 * import { AutoProcessor, AutoModelForAudioFrameClassification, read_audio } from '@huggingface/transformers';
 *
 * // Read and preprocess audio
 * const processor = await AutoProcessor.from_pretrained('Xenova/wavlm-base-plus-sd');
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav';
 * const audio = await read_audio(url, 16000);
 * const inputs = await processor(audio);
 *
 * // Run model with inputs
 * const model = await AutoModelForAudioFrameClassification.from_pretrained('Xenova/wavlm-base-plus-sd');
 * const { logits } = await model(inputs);
 * // {
 * //   logits: Tensor {
 * //     dims: [ 1, 549, 2 ],  // [batch_size, num_frames, num_speakers]
 * //     type: 'float32',
 * //     data: Float32Array(1098) [-3.5301010608673096, ...],
 * //     size: 1098
 * //   }
 * // }
 *
 * const labels = logits[0].sigmoid().tolist().map(
 *     frames => frames.map(speaker => speaker > 0.5 ? 1 : 0)
 * );
 * console.log(labels); // labels is a one-hot array of shape (num_frames, num_speakers)
 * // [
 * //     [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
 * //     [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
 * //     [0, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
 * //     ...
 * // ]
 * ```
 */
export class WavLMForAudioFrameClassification extends WavLMPreTrainedModel {
    /**
     * Calls the model on new inputs.
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<TokenClassifierOutput>} An object containing the model's output logits for sequence classification.
     */
    async _call(model_inputs) {
        return new TokenClassifierOutput(await super._call(model_inputs));
    }
}

export class StyleTextToSpeech2PreTrainedModel extends PreTrainedModel {}
export class StyleTextToSpeech2Model extends StyleTextToSpeech2PreTrainedModel {}

//////////////////////////////////////////////////
// SpeechT5 models
/**
 * An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained models.
 */
export class SpeechT5PreTrainedModel extends PreTrainedModel {}

/**
 * The bare SpeechT5 Encoder-Decoder Model outputting raw hidden-states without any specific pre- or post-nets.
 */
export class SpeechT5Model extends SpeechT5PreTrainedModel {}

/**
 * SpeechT5 Model with a speech encoder and a text decoder.
 *
 * **Example:** Generate speech from text with `SpeechT5ForSpeechToText`.
 * ```javascript
 * import { AutoTokenizer, AutoProcessor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, Tensor } from '@huggingface/transformers';
 *
 * // Load the tokenizer and processor
 * const tokenizer = await AutoTokenizer.from_pretrained('Xenova/speecht5_tts');
 * const processor = await AutoProcessor.from_pretrained('Xenova/speecht5_tts');
 *
 * // Load the models
 * // NOTE: We use the full-precision versions as they are more accurate
 * const model = await SpeechT5ForTextToSpeech.from_pretrained('Xenova/speecht5_tts', { dtype: 'fp32' });
 * const vocoder = await SpeechT5HifiGan.from_pretrained('Xenova/speecht5_hifigan', { dtype: 'fp32' });
 *
 * // Load speaker embeddings from URL
 * const speaker_embeddings_data = new Float32Array(
 *     await (await fetch('https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/speaker_embeddings.bin')).arrayBuffer()
 * );
 * const speaker_embeddings = new Tensor(
 *     'float32',
 *     speaker_embeddings_data,
 *     [1, speaker_embeddings_data.length]
 * )
 *
 * // Run tokenization
 * const { input_ids } = tokenizer('Hello, my dog is cute');
 *
 * // Generate waveform
 * const { waveform } = await model.generate_speech(input_ids, speaker_embeddings, { vocoder });
 * console.log(waveform)
 * // Tensor {
 * //   dims: [ 26112 ],
 * //   type: 'float32',
 * //   size: 26112,
 * //   data: Float32Array(26112) [ -0.00043630177970044315, -0.00018082228780258447, ... ],
 * // }
 * ```
 */
export class SpeechT5ForSpeechToText extends SpeechT5PreTrainedModel {}

/**
 * SpeechT5 Model with a text encoder and a speech decoder.
 */
export class SpeechT5ForTextToSpeech extends SpeechT5PreTrainedModel {
    /**
     * @typedef {Object} SpeechOutput
     * @property {Tensor} [spectrogram] The predicted log-mel spectrogram of shape
     * `(output_sequence_length, config.num_mel_bins)`. Returned when no `vocoder` is provided
     * @property {Tensor} [waveform] The predicted waveform of shape `(num_frames,)`. Returned when a `vocoder` is provided.
     * @property {Tensor} [cross_attentions] The outputs of the decoder's cross-attention layers of shape
     * `(config.decoder_layers, config.decoder_attention_heads, output_sequence_length, input_sequence_length)`. returned when `output_cross_attentions` is `true`.
     */

    /**
     * Converts a sequence of input tokens into a sequence of mel spectrograms, which are subsequently turned into a speech waveform using a vocoder.
     * @param {Tensor} input_values Indices of input sequence tokens in the vocabulary.
     * @param {Tensor} speaker_embeddings Tensor containing the speaker embeddings.
     * @param {Object} options Optional parameters for generating speech.
     * @param {number} [options.threshold=0.5] The generated sequence ends when the predicted stop token probability exceeds this value.
     * @param {number} [options.minlenratio=0.0] Used to calculate the minimum required length for the output sequence.
     * @param {number} [options.maxlenratio=20.0] Used to calculate the maximum allowed length for the output sequence.
     * @param {Object} [options.vocoder=null] The vocoder that converts the mel spectrogram into a speech waveform. If `null`, the output is the mel spectrogram.
     * @param {boolean} [options.output_cross_attentions=false] Whether or not to return the attentions tensors of the decoder's cross-attention layers.
     * @returns {Promise<SpeechOutput>} A promise which resolves to an object containing the spectrogram, waveform, and cross-attention tensors.
     */
    async generate_speech(
        input_values,
        speaker_embeddings,
        {
            threshold = 0.5,
            minlenratio = 0.0,
            maxlenratio = 20.0,
            vocoder = null,
            // output_cross_attentions = false, // TODO add
        } = {},
    ) {
        const model_inputs = {
            input_ids: input_values,
        };

        const { encoder_outputs, encoder_attention_mask } = await encoderForward(this, model_inputs);

        // @ts-expect-error TS2339
        const r = encoder_outputs.dims[1] / this.config.reduction_factor;
        const maxlen = Math.floor(r * maxlenratio);
        const minlen = Math.floor(r * minlenratio);

        // @ts-expect-error TS2339
        const num_mel_bins = this.config.num_mel_bins;

        let spectrogramParts = [];
        let past_key_values = null;
        let decoder_outputs = null;
        let idx = 0;

        while (true) {
            ++idx;

            const use_cache_branch = boolTensor(!!decoder_outputs);
            let output_sequence;
            if (decoder_outputs) {
                output_sequence = decoder_outputs.output_sequence_out;
            } else {
                output_sequence = new Tensor('float32', new Float32Array(num_mel_bins), [1, 1, num_mel_bins]);
            }
            let decoderFeeds = {
                use_cache_branch,
                output_sequence,
                encoder_attention_mask: encoder_attention_mask,
                speaker_embeddings: speaker_embeddings,
                encoder_hidden_states: encoder_outputs,
            };

            this.addPastKeyValues(decoderFeeds, past_key_values);
            decoder_outputs = await sessionRun(this.sessions['decoder_model_merged'], decoderFeeds);
            past_key_values = this.getPastKeyValues(decoder_outputs, past_key_values);

            const { prob, spectrum } = decoder_outputs;
            spectrogramParts.push(spectrum);

            if (
                idx >= minlen &&
                // Finished when stop token or maximum length is reached.
                (Array.from(prob.data).filter((p) => p >= threshold).length > 0 || idx >= maxlen)
            ) {
                break;
            }
        }

        const spectrogram = cat(spectrogramParts);
        const { waveform } = await sessionRun(vocoder.sessions['model'], { spectrogram });

        return {
            spectrogram,
            waveform,
            // cross_attentions: null, // TODO add
        };
    }
}

/**
 * HiFi-GAN vocoder.
 *
 * See [SpeechT5ForSpeechToText](./models#module_models.SpeechT5ForSpeechToText) for example usage.
 */
export class SpeechT5HifiGan extends PreTrainedModel {
    main_input_name = 'spectrogram';
}
//////////////////////////////////////////////////

export class SupertonicPreTrainedModel extends PreTrainedModel {}
export class SupertonicForConditionalGeneration extends SupertonicPreTrainedModel {
    async generate_speech({
        // Required inputs
        input_ids,
        attention_mask,
        style,

        // Optional inputs
        num_inference_steps = 5,
        speed = 1.05,
    }) {
        // @ts-expect-error TS2339
        const { sampling_rate, chunk_compress_factor, base_chunk_size, latent_dim } = this.config;

        // 1. Text Encoder
        const { last_hidden_state, durations } = await sessionRun(this.sessions['text_encoder'], {
            input_ids,
            attention_mask,
            style,
        });
        durations.div_(speed); // Apply speed factor to duration

        // 2. Latent Denoiser
        const wav_len_max = durations.max().item() * sampling_rate;
        const chunk_size = base_chunk_size * chunk_compress_factor;
        const latent_len = Math.floor((wav_len_max + chunk_size - 1) / chunk_size);
        const batch_size = input_ids.dims[0];
        const latent_mask = ones([batch_size, latent_len]);
        const num_steps = full([batch_size], num_inference_steps);

        let noisy_latents = randn([batch_size, latent_dim * chunk_compress_factor, latent_len]);
        for (let step = 0; step < num_inference_steps; ++step) {
            const timestep = full([batch_size], step);
            ({ denoised_latents: noisy_latents } = await sessionRun(this.sessions['latent_denoiser'], {
                style,
                noisy_latents,
                latent_mask,
                encoder_outputs: last_hidden_state,
                attention_mask,
                timestep,
                num_inference_steps: num_steps,
            }));
        }

        // 3. Voice Decoder
        const { waveform } = await sessionRun(this.sessions['voice_decoder'], {
            latents: noisy_latents,
        });
        return {
            waveform,
            durations,
        };
    }
}

//////////////////////////////////////////////////
// TrOCR models
export class TrOCRPreTrainedModel extends PreTrainedModel {}

/**
 * The TrOCR Decoder with a language modeling head.
 */
export class TrOCRForCausalLM extends TrOCRPreTrainedModel {}

//////////////////////////////////////////////////

//////////////////////////////////////////////////
// Mistral models
/**
 * The bare Mistral Model outputting raw hidden-states without any specific head on top.
 */
export class MistralPreTrainedModel extends PreTrainedModel {}

export class MistralModel extends MistralPreTrainedModel {}

export class MistralForCausalLM extends MistralPreTrainedModel {}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// ERNIE-4.5 models
export class Ernie4_5_PretrainedModel extends PreTrainedModel {}

export class Ernie4_5_Model extends Ernie4_5_PretrainedModel {}

export class Ernie4_5_ForCausalLM extends Ernie4_5_PretrainedModel {}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// Starcoder2 models
/**
 * The bare Starcoder2 Model outputting raw hidden-states without any specific head on top.
 */
export class Starcoder2PreTrainedModel extends PreTrainedModel {}

export class Starcoder2Model extends Starcoder2PreTrainedModel {}

export class Starcoder2ForCausalLM extends Starcoder2PreTrainedModel {}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// Falcon models
/**
 * The bare Falcon Model outputting raw hidden-states without any specific head on top.
 */
export class FalconPreTrainedModel extends PreTrainedModel {}

export class FalconModel extends FalconPreTrainedModel {}

export class FalconForCausalLM extends FalconPreTrainedModel {}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// CLAP models
export class ClapPreTrainedModel extends PreTrainedModel {}

export class ClapModel extends ClapPreTrainedModel {}

/**
 * CLAP Text Model with a projection layer on top (a linear layer on top of the pooled output).
 *
 * **Example:** Compute text embeddings with `ClapTextModelWithProjection`.
 *
 * ```javascript
 * import { AutoTokenizer, ClapTextModelWithProjection } from '@huggingface/transformers';
 *
 * // Load tokenizer and text model
 * const tokenizer = await AutoTokenizer.from_pretrained('Xenova/clap-htsat-unfused');
 * const text_model = await ClapTextModelWithProjection.from_pretrained('Xenova/clap-htsat-unfused');
 *
 * // Run tokenization
 * const texts = ['a sound of a cat', 'a sound of a dog'];
 * const text_inputs = tokenizer(texts, { padding: true, truncation: true });
 *
 * // Compute embeddings
 * const { text_embeds } = await text_model(text_inputs);
 * // Tensor {
 * //   dims: [ 2, 512 ],
 * //   type: 'float32',
 * //   data: Float32Array(1024) [ ... ],
 * //   size: 1024
 * // }
 * ```
 */
export class ClapTextModelWithProjection extends ClapPreTrainedModel {
    /** @type {typeof PreTrainedModel.from_pretrained} */
    static async from_pretrained(pretrained_model_name_or_path, options = {}) {
        return super.from_pretrained(pretrained_model_name_or_path, {
            ...options,
            // Update default model file name if not provided
            model_file_name: options.model_file_name ?? 'text_model',
        });
    }
}

/**
 * CLAP Audio Model with a projection layer on top (a linear layer on top of the pooled output).
 *
 * **Example:** Compute audio embeddings with `ClapAudioModelWithProjection`.
 *
 * ```javascript
 * import { AutoProcessor, ClapAudioModelWithProjection, read_audio } from '@huggingface/transformers';
 *
 * // Load processor and audio model
 * const processor = await AutoProcessor.from_pretrained('Xenova/clap-htsat-unfused');
 * const audio_model = await ClapAudioModelWithProjection.from_pretrained('Xenova/clap-htsat-unfused');
 *
 * // Read audio and run processor
 * const audio = await read_audio('https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cat_meow.wav');
 * const audio_inputs = await processor(audio);
 *
 * // Compute embeddings
 * const { audio_embeds } = await audio_model(audio_inputs);
 * // Tensor {
 * //   dims: [ 1, 512 ],
 * //   type: 'float32',
 * //   data: Float32Array(512) [ ... ],
 * //   size: 512
 * // }
 * ```
 */
export class ClapAudioModelWithProjection extends ClapPreTrainedModel {
    /** @type {typeof PreTrainedModel.from_pretrained} */
    static async from_pretrained(pretrained_model_name_or_path, options = {}) {
        return super.from_pretrained(pretrained_model_name_or_path, {
            ...options,
            // Update default model file name if not provided
            model_file_name: options.model_file_name ?? 'audio_model',
        });
    }
}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// VITS models
export class VitsPreTrainedModel extends PreTrainedModel {}

/**
 * The complete VITS model, for text-to-speech synthesis.
 *
 * **Example:** Generate speech from text with `VitsModel`.
 * ```javascript
 * import { AutoTokenizer, VitsModel } from '@huggingface/transformers';
 *
 * // Load the tokenizer and model
 * const tokenizer = await AutoTokenizer.from_pretrained('Xenova/mms-tts-eng');
 * const model = await VitsModel.from_pretrained('Xenova/mms-tts-eng');
 *
 * // Run tokenization
 * const inputs = tokenizer('I love transformers');
 *
 * // Generate waveform
 * const { waveform } = await model(inputs);
 * // Tensor {
 * //   dims: [ 1, 35328 ],
 * //   type: 'float32',
 * //   data: Float32Array(35328) [ ... ],
 * //   size: 35328,
 * // }
 * ```
 */
export class VitsModel extends VitsPreTrainedModel {
    /**
     * Calls the model on new inputs.
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<VitsModelOutput>} The outputs for the VITS model.
     */
    async _call(model_inputs) {
        return new VitsModelOutput(await super._call(model_inputs));
    }
}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// Segformer models
export class SegformerPreTrainedModel extends PreTrainedModel {}

/**
 * The bare SegFormer encoder (Mix-Transformer) outputting raw hidden-states without any specific head on top.
 */
export class SegformerModel extends SegformerPreTrainedModel {}

/**
 * SegFormer Model transformer with an image classification head on top (a linear layer on top of the final hidden states) e.g. for ImageNet.
 */
export class SegformerForImageClassification extends SegformerPreTrainedModel {}

/**
 * SegFormer Model transformer with an all-MLP decode head on top e.g. for ADE20k, CityScapes.
 */
export class SegformerForSemanticSegmentation extends SegformerPreTrainedModel {}

//////////////////////////////////////////////////

//////////////////////////////////////////////////
// StableLm models
export class StableLmPreTrainedModel extends PreTrainedModel {}

/**
 * The bare StableLm Model transformer outputting raw hidden-states without any specific head on top.
 */
export class StableLmModel extends StableLmPreTrainedModel {}

/**
 * StableLm Model with a `language modeling` head on top for Causal Language Modeling (with past).
 */
export class StableLmForCausalLM extends StableLmPreTrainedModel {}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
export class EfficientNetPreTrainedModel extends PreTrainedModel {}

/**
 * The bare EfficientNet model outputting raw features without any specific head on top.
 */
export class EfficientNetModel extends EfficientNetPreTrainedModel {}

/**
 * EfficientNet Model with an image classification head on top (a linear layer on top of the pooled features).
 */
export class EfficientNetForImageClassification extends EfficientNetPreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// Musicgen models
export class MusicgenPreTrainedModel extends PreTrainedModel {}

/**
 * The bare Musicgen decoder model outputting raw hidden-states without any specific head on top.
 */
export class MusicgenModel extends MusicgenPreTrainedModel {}

/**
 * The MusicGen decoder model with a language modelling head on top.
 */
export class MusicgenForCausalLM extends MusicgenPreTrainedModel {}

/**
 * The composite MusicGen model with a text encoder, audio encoder and Musicgen decoder,
 * for music generation tasks with one or both of text and audio prompts.
 *
 * **Example:** Generate music from text with `Xenova/musicgen-small`.
 * ```javascript
 * import { AutoTokenizer, MusicgenForConditionalGeneration } from '@huggingface/transformers';
 *
 * // Load tokenizer and model
 * const tokenizer = await AutoTokenizer.from_pretrained('Xenova/musicgen-small');
 * const model = await MusicgenForConditionalGeneration.from_pretrained(
 *   'Xenova/musicgen-small', { dtype: 'fp32' }
 * );
 *
 * // Prepare text input
 * const prompt = '80s pop track with bassy drums and synth';
 * const inputs = tokenizer(prompt);
 *
 * // Generate audio
 * const audio_values = await model.generate({
 *   ...inputs,
 *   max_new_tokens: 512,
 *   do_sample: true,
 *   guidance_scale: 3,
 * });
 *
 * // (Optional) Write the output to a WAV file
 * import wavefile from 'wavefile';
 * import fs from 'fs';
 *
 * const wav = new wavefile.WaveFile();
 * wav.fromScratch(1, model.config.audio_encoder.sampling_rate, '32f', audio_values.data);
 * fs.writeFileSync('musicgen_out.wav', wav.toBuffer());
 * ```
 */
export class MusicgenForConditionalGeneration extends PreTrainedModel {
    // NOTE: not MusicgenPreTrainedModel
    forward_params = [
        'input_ids',
        'attention_mask',
        'encoder_outputs',
        'decoder_input_ids',
        'decoder_attention_mask',
        'past_key_values',
    ];

    /**
     * Apply the pattern mask to the final ids,
     * then revert the pattern delay mask by filtering the pad token id in a single step.
     * @param {Tensor} outputs The output tensor from the model.
     * @returns {Tensor} The filtered output tensor.
     */
    _apply_and_filter_by_delay_pattern_mask(outputs) {
        const [bs_x_codebooks, seqLength] = outputs.dims;
        // @ts-expect-error TS2339
        const num_codebooks = this.config.decoder.num_codebooks;
        const upperBound = seqLength - num_codebooks;

        let newDataSize = 0;
        for (let i = 0; i < outputs.size; ++i) {
            // @ts-expect-error TS2339
            if (outputs.data[i] === this.config.decoder.pad_token_id) {
                continue;
            }

            const row = i % seqLength;
            const col = Math.floor(i / seqLength) % num_codebooks;

            const diff = row - col;
            if (diff > 0 && diff <= upperBound) {
                outputs.data[newDataSize++] = outputs.data[i];
            }
        }

        const batch_size = Math.floor(bs_x_codebooks / num_codebooks);
        const inferred = newDataSize / (batch_size * num_codebooks);
        // TODO: assert `inferred` is an integer
        return new Tensor(outputs.type, outputs.data.slice(0, newDataSize), [batch_size, num_codebooks, inferred]);
    }

    prepare_inputs_for_generation(input_ids, model_inputs, generation_config) {
        // apply the delay pattern mask
        let clonedInputIds = structuredClone(input_ids);
        for (let i = 0; i < clonedInputIds.length; ++i) {
            for (let j = 0; j < clonedInputIds[i].length; ++j) {
                // @ts-expect-error TS2339
                if (i % this.config.decoder.num_codebooks >= j) {
                    // @ts-expect-error TS2339
                    clonedInputIds[i][j] = BigInt(this.config.decoder.pad_token_id);
                }
            }
        }
        // for classifier free guidance we need to replicate the decoder args across the batch dim
        // (we'll split these before sampling)
        if (generation_config.guidance_scale !== null && generation_config.guidance_scale > 1) {
            // [batch, seqLength] -> [2 * batch, seqLength]
            clonedInputIds = clonedInputIds.concat(clonedInputIds);
        }

        const prepped = super.prepare_inputs_for_generation(clonedInputIds, model_inputs, generation_config);
        return prepped;
    }

    /**
     * Generates sequences of token ids for models with a language modeling head.
     * @param {import('./generation/parameters.js').GenerationFunctionParameters} options
     * @returns {Promise<ModelOutput|Tensor>} The output of the model, which can contain the generated token ids, attentions, and scores.
     */
    async generate(options) {
        const output_ids = await super.generate(options);

        // apply the pattern mask to the final ids
        // tensor: int64[1,batch_size,4,chunk_length]
        const audio_codes = this._apply_and_filter_by_delay_pattern_mask(/** @type {Tensor} */ (output_ids)).unsqueeze_(
            0,
        ); // append the frame dimension back to the audio codes

        const { audio_values } = await sessionRun(this.sessions['encodec_decode'], { audio_codes });

        return audio_values;
    }
}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// MobileNetV1 models
export class MobileNetV1PreTrainedModel extends PreTrainedModel {}

/**
 * The bare MobileNetV1 model outputting raw hidden-states without any specific head on top.
 */
export class MobileNetV1Model extends MobileNetV1PreTrainedModel {}

/**
 * MobileNetV1 model with an image classification head on top (a linear layer on top of the pooled features),
 * e.g. for ImageNet.
 */
export class MobileNetV1ForImageClassification extends MobileNetV1PreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}

export class MobileNetV1ForSemanticSegmentation extends MobileNetV1PreTrainedModel {}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// MobileNetV2 models
export class MobileNetV2PreTrainedModel extends PreTrainedModel {}

/**
 * The bare MobileNetV2 model outputting raw hidden-states without any specific head on top.
 */
export class MobileNetV2Model extends MobileNetV2PreTrainedModel {}

/**
 * MobileNetV2 model with an image classification head on top (a linear layer on top of the pooled features),
 * e.g. for ImageNet.
 */
export class MobileNetV2ForImageClassification extends MobileNetV2PreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
export class MobileNetV2ForSemanticSegmentation extends MobileNetV2PreTrainedModel {}
//////////////////////////////////////////////////

//////////////////////////////////////////////////
// MobileNetV3 models
export class MobileNetV3PreTrainedModel extends PreTrainedModel {}

/**
 * The bare MobileNetV3 model outputting raw hidden-states without any specific head on top.
 */
export class MobileNetV3Model extends MobileNetV3PreTrainedModel {}

/**
 * MobileNetV3 model with an image classification head on top (a linear layer on top of the pooled features),
 * e.g. for ImageNet.
 */
export class MobileNetV3ForImageClassification extends MobileNetV3PreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
export class MobileNetV3ForSemanticSegmentation extends MobileNetV3PreTrainedModel {}
//////////////////////////////////////////////////

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

export class MgpstrModelOutput extends ModelOutput {
    constructor({ char_logits, bpe_logits, wp_logits }) {
        super();
        this.char_logits = char_logits;
        this.bpe_logits = bpe_logits;
        this.wp_logits = wp_logits;
    }

    get logits() {
        return [this.char_logits, this.bpe_logits, this.wp_logits];
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

export class MimiEncoderOutput extends ModelOutput {
    /**
     * @param {Object} output The output of the model.
     * @param {Tensor} output.audio_codes Discrete code embeddings, of shape `(batch_size, num_quantizers, codes_length)`.
     */
    constructor({ audio_codes }) {
        super();
        this.audio_codes = audio_codes;
    }
}

export class MimiDecoderOutput extends ModelOutput {
    /**
     * @param {Object} output The output of the model.
     * @param {Tensor} output.audio_values Decoded audio values, of shape `(batch_size, num_channels, sequence_length)`.
     */
    constructor({ audio_values }) {
        super();
        this.audio_values = audio_values;
    }
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

export class DacEncoderOutput extends ModelOutput {
    /**
     * @param {Object} output The output of the model.
     * @param {Tensor} output.audio_codes Discrete code embeddings, of shape `(batch_size, num_quantizers, codes_length)`.
     */
    constructor({ audio_codes }) {
        super();
        this.audio_codes = audio_codes;
    }
}

export class DacDecoderOutput extends ModelOutput {
    /**
     * @param {Object} output The output of the model.
     * @param {Tensor} output.audio_values Decoded audio values, of shape `(batch_size, num_channels, sequence_length)`.
     */
    constructor({ audio_values }) {
        super();
        this.audio_values = audio_values;
    }
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

