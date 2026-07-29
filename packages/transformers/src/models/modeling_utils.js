import { Callable } from '../utils/generic.js';
import { constructSessions, sessionRun } from './session.js';
import { AutoConfig, getCacheNames } from '../configs.js';
import { Tensor, full_like, cat, zeros_like, ones_like, ones } from '../utils/tensor.js';
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
import { GenerationConfig } from '../generation/configuration_utils.js';
import {
    GenerationController,
    createLogitsProcessorList,
    createStoppingCriteriaList,
    prepareGenerationConfig,
    prepareGenerationLength,
} from '../generation/controller.js';
import { DefaultProgressCallback, pick } from '../utils/core.js';
import { ModelOutput } from './modeling_outputs.js';
import { logger } from '../utils/logger.js';
import { DynamicCache } from '../cache_utils.js';
import { get_model_files } from '../utils/model_registry/get_model_files.js';
import { get_file_metadata } from '../utils/model_registry/get_file_metadata.js';
import { MODEL_SESSION_CONFIG, MODEL_TYPES } from './session_config.js';
import { getModelId, isInferenceBackend, loadInferenceModel } from '../backends/inference.js';
import { getDefaultInferenceProvider, getOnnxProviderModule } from '../backends/default.js';

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

export { getSessionsConfig, getTextOnlySessions, MODEL_TYPES } from './session_config.js';

/**
 * Runtime-only model type configuration (forward functions, generation flags).
 * Session/file configuration lives in `MODEL_SESSION_CONFIG` (session_config.js)
 * and is merged in at lookup time by `resolveTypeConfig` to avoid duplication.
 */
const MODEL_RUNTIME_CONFIG = {
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
    [MODEL_TYPES.ImageAudioTextToText]: {
        can_generate: true,
        prepare_inputs: multimodal_text_to_text_prepare_inputs_for_generation,
    },
    [MODEL_TYPES.Phi3V]: {
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
    [MODEL_TYPES.VoxtralRealtime]: {
        can_generate: true,
        prepare_inputs: decoder_prepare_inputs_for_generation,
    },
    default: {
        can_generate: false,
        forward: encoder_forward,
    },
};

/**
 * Resolves the model type config for a given class name and config.
 * @param {string} modelName The name of the class being used to load.
 * @param {Object} config The model config.
 * @returns {{ typeConfig: Object, textOnly: boolean, modelType: number|undefined }}
 */
function resolveTypeConfig(modelName, config) {
    let modelType = MODEL_TYPE_MAPPING.get(modelName);
    let textOnly = false;

    // Detect cross-architecture loading: e.g., ForCausalLM class loading a ForConditionalGeneration model.
    // In this case, use the native architecture's type config (for forward/sessions) in text-only mode.
    const nativeArch = config?.architectures?.[0];
    if (
        nativeArch &&
        nativeArch !== modelName &&
        modelName?.endsWith('ForCausalLM') &&
        nativeArch.endsWith('ForConditionalGeneration')
    ) {
        const nativeType = MODEL_TYPE_MAPPING.get(nativeArch);
        if (nativeType !== undefined) {
            modelType = nativeType;
            textOnly = true;
        }
    }

    const runtimeConfig = MODEL_RUNTIME_CONFIG[modelType] ?? MODEL_RUNTIME_CONFIG.default;
    const sessionConfig = MODEL_SESSION_CONFIG[modelType] ?? MODEL_SESSION_CONFIG.default;
    return { typeConfig: { ...runtimeConfig, ...sessionConfig }, textOnly, modelType };
}

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
        const { typeConfig } = resolveTypeConfig(modelName, config);

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
     * @param {string|import('../backends/inference.js').InferenceBackend} pretrained_model_name_or_path The model backend, name, or path. A string selects the ONNX backend. It can be:
     * - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
     *   Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
     *   user or organization name, like `dbmdz/bert-base-german-cased`.
     * - A path to a *directory* containing model weights, e.g., `./my_model_directory/`.
     * @param {import('../utils/hub.js').PretrainedModelOptions} options Additional options for loading the model.
     *
     * @returns {Promise<PreTrainedModel>} A loaded model.
     */
    static async from_pretrained(pretrained_model_name_or_path, options = {}) {
        if (typeof pretrained_model_name_or_path === 'string') {
            const provider = await getDefaultInferenceProvider(pretrained_model_name_or_path);
            return provider.load({
                ...options,
                modelClass: this,
            });
        }
        if (typeof pretrained_model_name_or_path?.constructSessions === 'function') {
            if (pretrained_model_name_or_path.providerType === 'onnx') {
                await getOnnxProviderModule();
            }
            return /** @type {any} */ (pretrained_model_name_or_path).load({ ...options, modelClass: this });
        }
        if (isInferenceBackend(pretrained_model_name_or_path)) {
            const modelId = getModelId(pretrained_model_name_or_path);
            /** @type {import('../backends/inference.js').InferenceModelLoadOptions} */
            const resolvedOptions = { ...options };
            resolvedOptions.config =
                resolvedOptions.config ?? (await AutoConfig.from_pretrained(modelId, resolvedOptions));
            if (
                resolvedOptions.progress_callback &&
                !(resolvedOptions.progress_callback instanceof DefaultProgressCallback) &&
                typeof pretrained_model_name_or_path.listModelArtifacts === 'function'
            ) {
                const expectedFiles = await pretrained_model_name_or_path.listModelArtifacts({
                    ...resolvedOptions,
                    modelId,
                });
                const metadata = await Promise.all(
                    expectedFiles.map((file) =>
                        get_file_metadata(pretrained_model_name_or_path, file, resolvedOptions),
                    ),
                );
                /** @type {import('../utils/core.js').FilesLoadingMap} */
                const filesLoading = {};
                resolvedOptions.artifactMetadata = {};
                metadata.forEach((entry, index) => {
                    if (!entry.exists) return;
                    const file = expectedFiles[index];
                    resolvedOptions.artifactMetadata[file] = { size: entry.size, fromCache: entry.fromCache };
                    filesLoading[file] = {
                        loaded: entry.fromCache ? (entry.size ?? 0) : 0,
                        total: entry.size ?? 0,
                    };
                });
                resolvedOptions.progress_callback = new DefaultProgressCallback(
                    resolvedOptions.progress_callback,
                    filesLoading,
                );
            }
            // Custom models are duck-typed to the same runtime contract as PreTrainedModel.
            const model = /** @type {any} */ (await loadInferenceModel(pretrained_model_name_or_path, resolvedOptions));
            if (typeof model.createAutoregressiveSession === 'function' && model.generation_config == null) {
                model.generation_config = await getModelJSON(modelId, 'generation_config.json', false, resolvedOptions);
            }
            return model;
        }
        throw new TypeError('Unsupported pretrained model source.');
    }

    static async _from_pretrained(
        pretrained_model_name_or_path,
        {
            progress_callback = null,
            config = null,
            cache_dir = null,
            local_files_only = false,
            revision = 'main',
            model_file_name = null,
            subfolder = null,
            device = null,
            dtype = null,
            use_external_data_format = null,
            session_options = {},
            signal = undefined,
            artifactProvider = undefined,
            inferenceProvider = undefined,
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
            signal,
            artifactProvider,
            inferenceProvider,
        };

        const modelName = MODEL_CLASS_TO_NAME_MAPPING.get(this);

        config = options.config = await AutoConfig.from_pretrained(pretrained_model_name_or_path, options);

        const { typeConfig, textOnly, modelType } = resolveTypeConfig(modelName, config);

        if (modelType === undefined) {
            const type = modelName ?? config?.model_type;
            if (type !== 'custom') {
                logger.warn(
                    `Model type for '${type}' not found, assuming encoder-only architecture. Please report this at ${GITHUB_ISSUE_URL}.`,
                );
            }
        }

        // If a progress callback is provided AND it hasn't already been wrapped
        // by pipeline() (which does its own aggregation), gather file metadata
        // upfront so we can emit `progress_total` events. This lets consumers
        // render a single overall progress bar when calling from_pretrained() directly.
        if (progress_callback && !(progress_callback instanceof DefaultProgressCallback)) {
            /** @type {import('../utils/core.js').FilesLoadingMap} */
            const files_loading = {};

            try {
                const expected_files = await get_model_files(pretrained_model_name_or_path, {
                    config,
                    dtype,
                    device,
                    model_file_name,
                    inferenceProvider,
                });

                const metadata = await Promise.all(
                    expected_files.map((file) => get_file_metadata(pretrained_model_name_or_path, file, options)),
                );
                metadata.forEach((m, i) => {
                    if (m.exists) {
                        // config.json is fetched by AutoConfig.from_pretrained() above
                        const isAlreadyLoaded = expected_files[i] === 'config.json';
                        files_loading[expected_files[i]] = {
                            loaded: isAlreadyLoaded ? (m.size ?? 0) : 0,
                            total: m.size ?? 0,
                        };
                    }
                });
            } catch (e) {
                // If we fail to get metadata, we can still proceed without total progress.
                // This may happen with local-only models or custom cache setups.
                logger.warn(`Unable to fetch model file metadata for total progress tracking: ${e}`);
            }

            if (Object.keys(files_loading).length > 0) {
                options.progress_callback = new DefaultProgressCallback(progress_callback, files_loading);
            }
        }

        const sessions = typeConfig.sessions(config, options, textOnly);
        const promises = [constructSessions(sessions, options, typeConfig.cache_sessions)];
        if (typeConfig.optional_configs) {
            promises.push(get_optional_configs(pretrained_model_name_or_path, typeConfig.optional_configs, options));
        }
        const info = await Promise.all(promises);

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
     * @returns {import('../generation/logits_process.js').LogitsProcessorList}
     * @private
     */
    _get_logits_processor(
        generation_config,
        input_ids_seq_length,
        // encoder_input_ids, TODO
        // prefix_allowed_tokens_fn, TODO
        logits_processor = null,
    ) {
        return createLogitsProcessorList(generation_config, input_ids_seq_length, logits_processor);
    }

    /**
     * This function merges multiple generation configs together to form a final generation config to be used by the model for text generation.
     * It first creates an empty `GenerationConfig` object, then it applies the model's own `generation_config` property to it. Finally, if a `generation_config` object was passed in the arguments, it overwrites the corresponding properties in the final config with those of the passed config object.
     * @param {GenerationConfig|null} generation_config A `GenerationConfig` object containing generation parameters.
     * @param {Object} kwargs Additional generation parameters to be used in place of those in the `generation_config` object.
     * @returns {GenerationConfig} The final generation config object to be used by the model for text generation.
     */
    _prepare_generation_config(generation_config, kwargs, cls = GenerationConfig) {
        return prepareGenerationConfig({
            modelConfig: this.config,
            modelGenerationConfig: this.generation_config,
            generationConfig: generation_config,
            kwargs,
            configClass: cls,
        });
    }

    /**
     *
     * @param {GenerationConfig} generation_config
     * @param {import('../generation/stopping_criteria.js').StoppingCriteria|import('../generation/stopping_criteria.js').StoppingCriteria[]|import('../generation/stopping_criteria.js').StoppingCriteriaList} [stopping_criteria=null]
     */
    _get_stopping_criteria(generation_config, stopping_criteria = null) {
        return createStoppingCriteriaList(generation_config, this.config, stopping_criteria);
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
        model_inputs['past_key_values'] = getPastKeyValues(outputs, model_inputs.past_key_values);

        // update inputs for next run
        model_inputs['input_ids'] = new Tensor('int64', generated_input_ids.flat(), [generated_input_ids.length, 1]);

        if (!is_encoder_decoder) {
            // update attention mask
            model_inputs.attention_mask = cat(
                [model_inputs.attention_mask, ones([model_inputs.attention_mask.dims[0], 1])],
                1,
            );
        } else if ('decoder_attention_mask' in model_inputs) {
            model_inputs.decoder_attention_mask = cat(
                [model_inputs.decoder_attention_mask, ones([model_inputs.decoder_attention_mask.dims[0], 1])],
                1,
            );
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
     * @returns {{inputs_tensor: Tensor, model_inputs: Record<string, Tensor> & {past_key_values?: DynamicCache}, model_input_name: string}} The model-specific inputs for generation.
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

        model_inputs['decoder_attention_mask'] = ones_like(decoder_input_ids);

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
            model_kwargs: /** @type {Record<string, Tensor|number[]>} */ (kwargs),
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

        prepareGenerationLength(generation_config, input_ids_length);

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

        let attentions = {};
        let return_dict_items = {};
        const controller = new GenerationController({
            inputIds: input_ids,
            generationConfig: generation_config,
            logitsProcessor: prepared_logits_processor,
            stoppingCriteria: prepared_stopping_criteria,
            streamer,
            collectOutputs: (stepOutputs) => {
                if (!generation_config.return_dict_in_generate) return;
                if (generation_config.output_attentions) {
                    const token_attentions = getAttentions(stepOutputs);
                    for (const key in token_attentions) {
                        (attentions[key] ??= []).push(token_attentions[key]);
                    }
                } else if (this._return_dict_in_generate_keys) {
                    Object.assign(return_dict_items, pick(stepOutputs, this._return_dict_in_generate_keys));
                }
            },
        });

        if (controller.allDone) {
            return generation_config.return_dict_in_generate
                ? controller.finalize({ past_key_values: kwargs.past_key_values ?? new DynamicCache() })
                : controller.finalize();
        }

        let outputs;
        try {
            while (!controller.allDone) {
                // prepare model inputs
                model_inputs = this.prepare_inputs_for_generation(
                    controller.sequences,
                    model_inputs,
                    generation_config,
                );
                outputs = await this.forward(model_inputs);
                const step = await controller.step({ logits: outputs.logits, outputs });
                if (step.allDone) break;

                model_inputs = this._update_model_kwargs_for_generation({
                    generated_input_ids: step.generatedInputIds,
                    outputs,
                    model_inputs,
                    is_encoder_decoder,
                });
            }
        } catch (error) {
            controller.abort(error);
            throw error;
        }

        // Update past key values from the final forward pass
        const past_key_values = getPastKeyValues(outputs, model_inputs.past_key_values);

        // Dispose output tensors not held by the cache
        const cachedTensors = new Set(Object.values(past_key_values));
        for (const tensor of Object.values(outputs)) {
            if (tensor.location === 'gpu-buffer' && !cachedTensors.has(tensor)) {
                tensor.dispose();
            }
        }

        // Dispose cache tensors if no one needs them
        const keepCacheAlive = 'past_key_values' in kwargs || generation_config.return_dict_in_generate;
        if (!keepCacheAlive) {
            await past_key_values.dispose();
        }

        if (generation_config.return_dict_in_generate) {
            return controller.finalize({
                past_key_values,
                ...attentions,
                ...return_dict_items,
                // TODO:
                // scores,
                // logits,
            });
        }
        return controller.finalize();
    }

    /**
     * Helper function to select valid inputs and run through the appropriate encoder (vision, text, audio) based on the input type.
     * @param {string} sessionName
     * @param {Record<string, Tensor>} inputs
     * @param {string} outputName
     * @private
     */
    async _encode_input(sessionName, inputs, outputName) {
        if (!Object.hasOwn(this.sessions, sessionName)) {
            throw new Error(`Model does not have a ${sessionName} session.`);
        }
        const session = this.sessions[sessionName];
        const output = await sessionRun(session, pick(inputs, session.inputNames));
        return output[outputName];
    }

    async encode_image(inputs) {
        return this._encode_input('vision_encoder', inputs, 'image_features');
    }

    async encode_text(inputs) {
        return this._encode_input('embed_tokens', inputs, 'inputs_embeds');
    }

    async encode_audio(inputs) {
        return this._encode_input('audio_encoder', inputs, 'audio_features');
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
    let { encoder_outputs, input_ids, decoder_input_ids, decoder_attention_mask, ...other_decoder_inputs } =
        model_inputs;
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

    // Pass decoder_attention_mask as attention_mask to the decoder session
    if (decoder_attention_mask && !other_decoder_inputs.attention_mask) {
        other_decoder_inputs.attention_mask = decoder_attention_mask;
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
 * Returns a DynamicCache containing past key values from the given decoder results object.
 * Always updates in-place when pastKeyValues is provided; creates a new DynamicCache otherwise.
 *
 * @param {Object} decoderResults The decoder results object.
 * @param {DynamicCache} pastKeyValues The previous past key values.
 * @returns {DynamicCache} The updated past key values cache.
 */
export function getPastKeyValues(decoderResults, pastKeyValues) {
    /** @type {Record<string, Tensor>} */
    const pkvs = Object.create(null);

    for (const name in decoderResults) {
        if (name.startsWith('present')) {
            const newName = name
                // Hybrid cache architecture
                .replace('present_ssm', 'past_ssm') // Mamba
                .replace('present_conv', 'past_conv') // LFM2
                .replace('present_recurrent', 'past_recurrent') // Qwen3.5
                .replace('present_compressor', 'past_compressor') // Deepseek V4
                .replace('present_indexer', 'past_indexer') // Deepseek V4

                // Standard cache architecture
                .replace('present', 'past_key_values');
            const is_encoder_pkv = name.includes('encoder');
            if (is_encoder_pkv && pastKeyValues) {
                // Optimization introduced by optimum to reuse past key values.
                // So, we just replace the constant outputs (`decoderResults[name]`) with the previous past key values.
                // https://github.com/huggingface/optimum/blob/0bf2c05fb7e1182b52d21b703cfc95fd9e4ea3dc/optimum/onnxruntime/base.py#L677-L704
                pkvs[newName] = pastKeyValues[newName];
            } else {
                pkvs[newName] = decoderResults[name];
            }
        }
    }

    if (pastKeyValues) {
        pastKeyValues.update(pkvs);
        return pastKeyValues;
    }
    return new DynamicCache(pkvs);
}

/**
 * Returns an object containing attentions from the given model output object.
 *
 * @param {Object} model_output The output of the model.
 * @returns {{cross_attentions?: Tensor[]}} An object containing attentions.
 */
export function getAttentions(model_output) {
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
 * Resolve symbolic dims from ONNX inputMetadata for empty-cache initialization.
 * Each symbolic dim name is looked up in `symbols`; numeric dims pass through.
 * Any unresolved symbolic dim defaults to 0.
 * @param {ReadonlyArray<number|string>} metadataShape
 * @param {Record<string, number>} symbols
 * @returns {number[]}
 */
export function resolveCacheShape(metadataShape, symbols) {
    return metadataShape.map((d) => {
        if (typeof d === 'number') return d;
        return symbols[d] ?? 0;
    });
}

/**
 * Adds past key values to the decoder feeds object. If pastKeyValues is null,
 * creates a new DynamicCache with zero-filled tensors for each cache entry.
 *
 * @param {PreTrainedModel} self The model instance.
 * @param {Record<string, any>} decoderFeeds The decoder feeds object to add past key values to.
 * @param {DynamicCache|null} pastKeyValues The cache containing past key values.
 * @returns {DynamicCache} The past key values cache (existing or newly created).
 */
export function addPastKeyValues(self, decoderFeeds, pastKeyValues) {
    if (pastKeyValues && Object.keys(pastKeyValues).length > 0) {
        Object.assign(decoderFeeds, pastKeyValues);
        return pastKeyValues;
    }

    const session = self.sessions['decoder_model_merged'] ?? self.sessions['model'];
    const batch_size = (decoderFeeds[self.main_input_name] ?? decoderFeeds.attention_mask)?.dims?.[0] ?? 1;

    const names = getCacheNames(self.config);
    const num_heads = self.config?.normalized_config?.num_heads;
    /** @type {Record<string, number>} */
    const symbols = { batch_size };
    if (typeof num_heads === 'number') {
        symbols['batch_size x num_heads'] = batch_size * num_heads;
    }
    /** @type {Record<string, Tensor>} */
    const entries = Object.create(null);
    for (const meta of session.inputMetadata) {
        if (!names.has(meta.name)) continue;
        const shape = resolveCacheShape(meta.shape, symbols);
        const size = shape.reduce((a, b) => a * b, 1);
        const cls = DataTypeMap[meta.type];
        const t = new Tensor(meta.type, new cls(size), shape);
        decoderFeeds[meta.name] = t;
        entries[meta.name] = t;
    }
    if (pastKeyValues) {
        // Populate the (empty) user-provided cache in-place
        pastKeyValues.update(entries);
        return pastKeyValues;
    }
    return new DynamicCache(entries);
}

/**
 * Sets `num_logits_to_keep` on `model_inputs` if the decoder session declares it as an input
 * and it has not already been set.
 *
 * `num_logits_to_keep` specifies how many trailing prompt logits the model computes:
 * - `0n` (or unset) computes logits for the entire sequence — used for prefill/scoring.
 * - `1n` computes only the last token's logits — used during autoregressive generation,
 *   since only the last prompt token's logits are needed to sample the next token. For long
 *   sequences, computing all logits uses a lot of memory, so `1n` significantly reduces the
 *   memory footprint.
 * - Any other positive integer keeps the last `num_logits_to_keep` logits.
 *
 * @param {PreTrainedModel} self The model instance.
 * @param {Record<string, any>} model_inputs The model inputs to mutate.
 * @param {bigint} value The value to set (typically `1n` for generation, `0n` as a fallback).
 * @private
 */
export function setNumLogitsToKeep(self, model_inputs, value) {
    if (model_inputs.num_logits_to_keep) return;
    const session = self.sessions['decoder_model_merged'] ?? self.sessions['model'];
    if (session?.inputNames.includes('num_logits_to_keep')) {
        model_inputs.num_logits_to_keep = new Tensor('int64', [value], []);
    }
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
        new_model_inputs.use_cache_branch = boolTensor(
            past_key_values != null && Object.keys(past_key_values).length > 0,
        );
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

    // Fallback for non-generation forward calls (e.g. prefill scoring): compute all logits.
    setNumLogitsToKeep(self, new_model_inputs, 0n);

    // Unpack the `past_key_values` object into model inputs
    addPastKeyValues(self, new_model_inputs, past_key_values);

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
 * @param {string[]} [params.modality_input_names] The modality input name.
 * @param {string} [params.modality_output_name] The modality output name.
 * @param {Tensor} [params.input_ids=null]
 * @param {Tensor} [params.attention_mask=null]
 * @param {Tensor} [params.position_ids=null]
 * @param {Tensor} [params.inputs_embeds=null]
 * @param {DynamicCache} [params.past_key_values=null]
 * @param {Object} [params.generation_config=null]
 * @param {Object} [params.logits_processor=null]
 * @param {Tensor} [params.num_logits_to_keep=null]
 * @returns {Promise<Tensor>} The model's output tensor
 * @private
 */
export async function generic_text_to_text_forward(
    self,
    {
        // Generic parameters:
        encode_function,
        merge_function,
        modality_input_names,
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
        num_logits_to_keep = null,

        // Additional parameters
        ...kwargs
    },
) {
    if (!inputs_embeds) {
        // 1. Extract the text embeddings.
        inputs_embeds = await self.encode_text({ input_ids, ...kwargs });

        // 2. Possibly, merge text and modality values
        const modality_values = pick(kwargs, modality_input_names);
        if (Object.keys(modality_values).length > 0) {
            if (input_ids.dims[1] !== 1) {
                const modality_features = await encode_function({
                    // Pass the modality values under its expected key.
                    // The caller knows whether this is audio or image.
                    ...modality_values,
                    ...kwargs,
                });
                ({ inputs_embeds, attention_mask } = merge_function({
                    [modality_output_name]: modality_features,
                    inputs_embeds,
                    input_ids,
                    attention_mask,
                }));
            } else if (past_key_values && input_ids.dims[1] === 1) {
                // This branch handles the cache case.
                const target_length = input_ids.dims[1]; // always 1
                const past_length = past_key_values.get_seq_length();

                attention_mask = cat(
                    [
                        ones([input_ids.dims[0], past_length]),
                        attention_mask.slice(null, [attention_mask.dims[1] - target_length, attention_mask.dims[1]]),
                    ],
                    1,
                );
            }
        }
    }

    if (!position_ids) {
        if (
            // Handle special case for qwen vl models
            [
                'qwen2_vl',
                'qwen2_vl_text',
                'qwen2_5_vl',
                'qwen2_5_vl_text',
                'qwen3_vl',
                'qwen3_vl_text',
                'qwen3_vl_moe',
                'qwen3_vl_moe_text',
                'qwen3_5',
                'qwen3_5_text',
                'qwen3_5_moe',
                'qwen3_5_moe_text',
                'glm_ocr',
                'glm_ocr_text',
            ].includes(self.config.model_type)
        ) {
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
            num_logits_to_keep,
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
        modality_input_names: ['audio_values', 'input_features'],
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
        modality_input_names: ['pixel_values'],
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
    const past_length = model_inputs.past_key_values ? model_inputs.past_key_values.get_seq_length() : 0;

    setNumLogitsToKeep(self, model_inputs, 1n);

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

    setNumLogitsToKeep(self, model_inputs, 1n);

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

    // Currently, we require modality features to be in float32 for correct scatter behavior.
    // TODO: In future, detect dtype of embeds and cast modality features to the same dtype before scattering.
    // modality_features = modality_features.to('float32');

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
