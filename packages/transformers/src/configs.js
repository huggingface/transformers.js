/**
 * @file Helper module for using model configs. For more information, see the corresponding
 * [Python documentation](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoConfig).
 *
 * **Example:** Load an `AutoConfig`.
 *
 * ```javascript
 * import { AutoConfig } from '@huggingface/transformers';
 * const config = await AutoConfig.from_pretrained('bert-base-uncased');
 * console.log(config);
 * // PretrainedConfig {
 * //   "model_type": "bert",
 * //   "is_encoder_decoder": false,
 * //   "architectures": [
 * //       "BertForMaskedLM"
 * //   ],
 * //   "vocab_size": 30522
 * //   "num_attention_heads": 12,
 * //   "num_hidden_layers": 12,
 * //   "hidden_size": 768,
 * //   "max_position_embeddings": 512,
 * //   ...
 * // }
 * ```
 *
 * @module configs
 */

import { pick } from './utils/core.js';
import { getModelJSON } from './utils/hub.js';

/**
 * @typedef {import('./utils/hub.js').PretrainedOptions} PretrainedOptions
 */

/**
 * @typedef {import('./utils/core.js').ProgressCallback} ProgressCallback
 */

/**
 * @typedef {import('./utils/core.js').ProgressInfo} ProgressInfo
 */

/**
 * Loads a config from the specified path.
 * @param {string} pretrained_model_name_or_path The path to the config directory.
 * @param {PretrainedOptions} options Additional options for loading the config.
 * @returns {Promise<Object>} A promise that resolves with information about the loaded config.
 */
async function loadConfig(pretrained_model_name_or_path, options) {
    return await getModelJSON(pretrained_model_name_or_path, 'config.json', true, options);
}

/**
 *
 * @param {PretrainedConfig} config
 * @returns {Object} The normalized configuration.
 */
function getNormalizedConfig(config) {
    const mapping = {};

    let init_normalized_config = {};
    switch (config.model_type) {
        // Sub-configs
        case 'llava':
        case 'paligemma':
        case 'gemma3':
        case 'florence2':
        case 'llava_onevision':
        case 'idefics3':
        case 'granite_speech':
        case 'ultravox':
        case 'voxtral':
        case 'voxtral_realtime':
        case 'smolvlm':
        case 'gemma3n':
        case 'gemma4':
        case 'lfm2_vl':
        case 'chatterbox':
        case 'lighton_ocr':
        case 'glm_ocr':
        case 'mistral3':
        case 'qwen2_5_vl':
        case 'qwen3_vl':
        case 'qwen3_vl_moe':
            // @ts-expect-error TS2339
            init_normalized_config = getNormalizedConfig(config.text_config);
            break;
        case 'moondream1':
            // @ts-expect-error TS2339
            init_normalized_config = getNormalizedConfig(config.phi_config);
            break;
        case 'musicgen':
            // @ts-expect-error TS2339
            init_normalized_config = getNormalizedConfig(config.decoder);
            break;
        case 'multi_modality':
            // @ts-expect-error TS2339
            init_normalized_config = getNormalizedConfig(config.language_config);
            break;

        // Decoder-only models
        case 'gpt2':
        case 'gptj':
        case 'jais':
        case 'codegen':
        case 'gpt_bigcode':
            mapping['num_heads'] = 'n_head';
            mapping['num_layers'] = 'n_layer';
            mapping['hidden_size'] = 'n_embd';
            break;
        case 'gpt_neox':
        case 'stablelm':
        case 'opt':
        case 'falcon':
        case 'modernbert-decoder':
            mapping['num_heads'] = 'num_attention_heads';
            mapping['num_layers'] = 'num_hidden_layers';
            mapping['hidden_size'] = 'hidden_size';
            break;
        case 'gpt_oss':
        case 'llama':
        case 'llama4_text':
        case 'nanochat':
        case 'apertus':
        case 'arcee':
        case 'afmoe':
        case 'lfm2':
        case 'lfm2_moe':
        case 'smollm3':
        case 'olmo':
        case 'olmo2':
        case 'olmo3':
        case 'mobilellm':
        case 'granite':
        case 'granitemoehybrid':
        case 'cohere':
        case 'cohere2':
        case 'mistral':
        case 'voxtral_realtime_text':
        case 'voxtral_realtime_encoder':
        case 'starcoder2':
        case 'qwen2':
        case 'qwen2_moe':
        case 'qwen2_vl':
        case 'qwen2_vl_text':
        case 'qwen2_5_vl_text':
        case 'qwen3_moe':
        case 'qwen3_vl_text':
        case 'qwen3_vl_moe_text':
        case 'phi':
        case 'phi3':
        case 'phi3_v':
        case 'llava_qwen2':
            mapping['num_heads'] = 'num_key_value_heads';
            mapping['num_layers'] = 'num_hidden_layers';
            mapping['hidden_size'] = 'hidden_size';
            mapping['num_attention_heads'] = 'num_attention_heads';
            mapping['dim_kv'] = 'head_dim';
            break;
        case 'qwen3':
        case 'solar_open':
        case 'glm_ocr_text':
        case 'gemma':
        case 'gemma2':
        case 'vaultgemma':
        case 'gemma3_text':
        case 'gemma3n_text':
        case 'gemma4_text':
        case 'glm':
        case 'helium':
        case 'ernie4_5':
        case 'hunyuan_v1_dense':
        case 'falcon_h1':
        case 'nemotron_h':
        case 'ministral':
        case 'ministral3':
            mapping['num_heads'] = 'num_key_value_heads';
            mapping['num_layers'] = 'num_hidden_layers';
            mapping['dim_kv'] = 'head_dim';
            break;
        case 'openelm':
            mapping['num_heads'] = 'num_kv_heads';
            mapping['num_layers'] = 'num_transformer_layers';
            mapping['dim_kv'] = 'head_dim';
            break;
        case 'gpt_neo':
        case 'donut-swin':
            mapping['num_heads'] = 'num_heads';
            mapping['num_layers'] = 'num_layers';
            mapping['hidden_size'] = 'hidden_size';
            break;
        case 'bloom':
            mapping['num_heads'] = 'n_head';
            mapping['num_layers'] = 'n_layer';
            mapping['hidden_size'] = 'hidden_size';
            break;
        case 'mpt':
            mapping['num_heads'] = 'n_heads';
            mapping['num_layers'] = 'n_layers';
            mapping['hidden_size'] = 'd_model';
            break;
        case 'exaone':
            mapping['num_heads'] = 'num_key_value_heads';
            mapping['num_layers'] = 'num_layers';
            mapping['dim_kv'] = 'head_dim';
            mapping['num_attention_heads'] = 'num_attention_heads';
            break;
        case 'youtu':
        case 'deepseek_v3':
        case 'glm_moe_dsa':
        case 'mistral4':
            mapping['num_heads'] = 'num_key_value_heads';
            mapping['num_layers'] = 'num_hidden_layers';
            mapping['dim_kv'] = 'qk_head_dim';
            mapping['num_attention_heads'] = 'num_attention_heads';
            break;

        // Encoder-decoder models
        case 't5':
        case 'mt5':
        case 'longt5':
            mapping['num_decoder_layers'] = 'num_decoder_layers';
            mapping['num_decoder_heads'] = 'num_heads';
            mapping['decoder_dim_kv'] = 'd_kv';
            mapping['num_encoder_layers'] = 'num_layers';
            mapping['num_encoder_heads'] = 'num_heads';
            mapping['encoder_dim_kv'] = 'd_kv';
            break;
        case 'bart':
        case 'mbart':
        case 'marian':
        case 'whisper':
        case 'lite-whisper':
        case 'm2m_100':
        case 'blenderbot':
        case 'blenderbot-small':
        case 'florence2_language':
            mapping['num_decoder_layers'] = 'decoder_layers';
            mapping['num_decoder_heads'] = 'decoder_attention_heads';
            mapping['decoder_hidden_size'] = 'd_model';
            mapping['num_encoder_layers'] = 'encoder_layers';
            mapping['num_encoder_heads'] = 'encoder_attention_heads';
            mapping['encoder_hidden_size'] = 'd_model';
            break;
        case 'speecht5':
            mapping['num_decoder_layers'] = 'decoder_layers';
            mapping['num_decoder_heads'] = 'decoder_attention_heads';
            mapping['decoder_hidden_size'] = 'hidden_size';
            mapping['num_encoder_layers'] = 'encoder_layers';
            mapping['num_encoder_heads'] = 'encoder_attention_heads';
            mapping['encoder_hidden_size'] = 'hidden_size';
            break;
        case 'trocr':
            mapping['num_encoder_layers'] = mapping['num_decoder_layers'] = 'decoder_layers';
            mapping['num_encoder_heads'] = mapping['num_decoder_heads'] = 'decoder_attention_heads';
            mapping['encoder_hidden_size'] = mapping['decoder_hidden_size'] = 'd_model';
            break;
        case 'musicgen_decoder':
            mapping['num_encoder_layers'] = mapping['num_decoder_layers'] = 'num_hidden_layers';
            mapping['num_encoder_heads'] = mapping['num_decoder_heads'] = 'num_attention_heads';
            mapping['encoder_hidden_size'] = mapping['decoder_hidden_size'] = 'hidden_size';
            break;
        case 'moonshine':
            mapping['num_decoder_layers'] = 'decoder_num_hidden_layers';
            mapping['num_decoder_heads'] = 'decoder_num_key_value_heads';
            mapping['num_encoder_layers'] = 'encoder_num_hidden_layers';
            mapping['num_encoder_heads'] = 'encoder_num_key_value_heads';
            mapping['encoder_hidden_size'] = mapping['decoder_hidden_size'] = 'hidden_size';
            break;
        case 'cohere_asr':
            mapping['num_decoder_layers'] = 'num_hidden_layers';
            mapping['num_decoder_heads'] = 'num_key_value_heads';
            mapping['decoder_hidden_size'] = 'hidden_size';
            mapping['decoder_dim_kv'] = 'head_dim';
            const {
                num_hidden_layers: num_encoder_layers,
                num_attention_heads: num_encoder_heads,
                hidden_size: encoder_hidden_size,
            } = /** @type {any} */ (config).encoder_config;
            init_normalized_config = {
                num_encoder_layers,
                num_encoder_heads,
                encoder_hidden_size,
                // @ts-expect-error TS2339
                encoder_dim_kv: config.head_dim,
            };
            break;
        case 'vision-encoder-decoder':
            // @ts-expect-error TS2339
            const decoderConfig = getNormalizedConfig(config.decoder);

            const add_encoder_pkv = 'num_decoder_layers' in decoderConfig;
            const result = pick(config, ['model_type', 'is_encoder_decoder']);
            if (add_encoder_pkv) {
                // Decoder is part of an encoder-decoder model
                result.num_decoder_layers = decoderConfig.num_decoder_layers;
                result.num_decoder_heads = decoderConfig.num_decoder_heads;
                result.decoder_hidden_size = decoderConfig.decoder_hidden_size;

                result.num_encoder_layers = decoderConfig.num_encoder_layers;
                result.num_encoder_heads = decoderConfig.num_encoder_heads;
                result.encoder_hidden_size = decoderConfig.encoder_hidden_size;
            } else {
                // Decoder is a decoder-only model
                result.num_layers = decoderConfig.num_layers;
                result.num_heads = decoderConfig.num_heads;
                result.hidden_size = decoderConfig.hidden_size;
            }
            return result;
    }

    // NOTE: If `num_attention_heads` is not set, it is assumed to be equal to `num_heads`
    const normalized_config = {
        ...init_normalized_config,
        ...pick(config, ['model_type', 'multi_query', 'is_encoder_decoder']),
    };
    for (const key in mapping) {
        normalized_config[key] = config[mapping[key]];
    }
    return normalized_config;
}

/**
 * @param {PretrainedConfig} config
 * @param {{ prefix?: string, session_name?: string }} [options]
 * @returns {Set<string>}
 */
export function getCacheNames(config, options) {
    if (!(config instanceof PretrainedConfig)) {
        config = new PretrainedConfig(config);
    }

    const pkv_prefix = options?.prefix ?? 'past_key_values';
    const conv_prefix = pkv_prefix === 'present' ? 'present' : 'past';
    /** @type {Set<string>} */
    const names = new Set();

    if (['lfm2', 'lfm2_moe'].includes(config.model_type)) {
        const { layer_types } = /** @type {any} */ (config);
        for (let i = 0; i < layer_types.length; ++i) {
            if (layer_types[i] === 'full_attention') {
                names.add(`${pkv_prefix}.${i}.key`);
                names.add(`${pkv_prefix}.${i}.value`);
            } else if (layer_types[i] === 'conv') {
                names.add(`${conv_prefix}_conv.${i}`);
            } else {
                throw new Error(`Unsupported layer type: ${layer_types[i]}`);
            }
        }
        return names;
    } else if (['granitemoehybrid', 'falcon_h1', 'nemotron_h'].includes(config.model_type)) {
        const c = /** @type {any} */ (config);
        const layer_types = c.layer_types ?? c.layers_block_type;
        const num_layers = c.num_hidden_layers ?? layer_types?.length;

        for (let i = 0; i < num_layers; ++i) {
            if (!layer_types || layer_types[i] === 'mamba') {
                names.add(`${conv_prefix}_conv.${i}`);
                names.add(`${conv_prefix}_ssm.${i}`);
            }
            if (!layer_types || layer_types[i] === 'attention') {
                names.add(`${pkv_prefix}.${i}.key`);
                names.add(`${pkv_prefix}.${i}.value`);
            }
        }
        return names;
    } else if (['qwen3_next', 'qwen3_5_text', 'qwen3_5_moe_text', 'olmo_hybrid'].includes(config.model_type)) {
        const { layer_types } = /** @type {any} */ (config);
        for (let i = 0; i < layer_types.length; ++i) {
            if (layer_types[i] === 'full_attention') {
                names.add(`${pkv_prefix}.${i}.key`);
                names.add(`${pkv_prefix}.${i}.value`);
            } else if (layer_types[i] === 'linear_attention') {
                if (config.model_type === 'olmo_hybrid') {
                    names.add(`${conv_prefix}_conv.${i}.key`);
                    names.add(`${conv_prefix}_conv.${i}.value`);
                    names.add(`${conv_prefix}_conv.${i}.query`);
                } else {
                    names.add(`${conv_prefix}_conv.${i}`);
                }
                names.add(`${conv_prefix}_recurrent.${i}`);
            } else {
                throw new Error(`Unsupported layer type: ${layer_types[i]}`);
            }
        }
        return names;
    } else if (['gemma4', 'gemma4_text'].includes(config.model_type)) {
        const c = /** @type {any} */ (
            config.model_type === 'gemma4' ? /** @type {any} */ (config).text_config : config
        );
        const num_hidden_layers = c.num_hidden_layers;
        const num_kv_shared_layers = c.num_kv_shared_layers ?? 0;
        const num_kv_layers = num_hidden_layers - num_kv_shared_layers;

        for (let i = 0; i < num_kv_layers; ++i) {
            names.add(`${pkv_prefix}.${i}.key`);
            names.add(`${pkv_prefix}.${i}.value`);
        }
        return names;
    } else if (['lfm2_vl', 'qwen3_5', 'qwen3_5_moe', 'voxtral_realtime'].includes(config.model_type)) {
        let subConfig;
        if (config.model_type === 'voxtral_realtime' && options?.session_name === 'audio_encoder') {
            subConfig = /** @type {any} */ (config).audio_config;
        } else {
            subConfig = /** @type {any} */ (config).text_config;
        }
        return getCacheNames(subConfig, options);
    }

    return getKeyValueNames(config, { prefix: pkv_prefix });
}

/**
 * @param {PretrainedConfig} config
 * @param {{ prefix?: string }} [options]
 * @returns {Set<string>}
 */
function getKeyValueNames(config, { prefix = 'past_key_values' } = {}) {
    /** @type {Set<string>} */
    const names = new Set();
    const normalized_config = config.normalized_config;

    if (
        normalized_config.is_encoder_decoder &&
        'num_encoder_heads' in normalized_config &&
        'num_decoder_heads' in normalized_config
    ) {
        for (let i = 0; i < normalized_config.num_decoder_layers; ++i) {
            names.add(`${prefix}.${i}.encoder.key`);
            names.add(`${prefix}.${i}.encoder.value`);
            names.add(`${prefix}.${i}.decoder.key`);
            names.add(`${prefix}.${i}.decoder.value`);
        }
    } else if (normalized_config.multi_query) {
        // e.g., for `gpt_bigcode`
        for (let i = 0; i < normalized_config.num_layers; ++i) {
            names.add(`${prefix}.${i}.key_value`);
        }
    } else {
        for (let i = 0; i < normalized_config.num_layers; ++i) {
            names.add(`${prefix}.${i}.key`);
            names.add(`${prefix}.${i}.value`);
        }
    }

    return names;
}
/**
 * Base class for all configuration classes. For more information, see the corresponding
 * [Python documentation](https://huggingface.co/docs/transformers/main/en/main_classes/configuration#transformers.PretrainedConfig).
 */
export class PretrainedConfig {
    // NOTE: Typo in original

    /** @type {string|null} */
    model_type = null;

    /** @type {boolean} */
    is_encoder_decoder = false;

    /** @type {number} */
    max_position_embeddings;

    /** @type {TransformersJSConfig} */
    'transformers.js_config';

    /**
     * Create a new PreTrainedTokenizer instance.
     * @param {Object} configJSON The JSON of the config.
     */
    constructor(configJSON) {
        Object.assign(this, configJSON);
        this.normalized_config = getNormalizedConfig(this);
    }

    /**
     * Loads a pre-trained config from the given `pretrained_model_name_or_path`.
     *
     * @param {string} pretrained_model_name_or_path The path to the pre-trained config.
     * @param {PretrainedOptions} options Additional options for loading the config.
     * @throws {Error} Throws an error if the config.json is not found in the `pretrained_model_name_or_path`.
     *
     * @returns {Promise<PretrainedConfig>} A new instance of the `PretrainedConfig` class.
     */
    static async from_pretrained(
        pretrained_model_name_or_path,
        { progress_callback = null, config = null, cache_dir = null, local_files_only = false, revision = 'main' } = {},
    ) {
        if (config && !(config instanceof PretrainedConfig)) {
            config = new PretrainedConfig(config);
        }

        const data =
            config ??
            (await loadConfig(pretrained_model_name_or_path, {
                progress_callback,
                config,
                cache_dir,
                local_files_only,
                revision,
            }));
        return new this(data);
    }
}

/**
 * Helper class which is used to instantiate pretrained configs with the `from_pretrained` function.
 *
 * @example
 * const config = await AutoConfig.from_pretrained('Xenova/bert-base-uncased');
 */
export class AutoConfig {
    /** @type {typeof PretrainedConfig.from_pretrained} */
    static async from_pretrained(...args) {
        return PretrainedConfig.from_pretrained(...args);
    }
}

/**
 * Transformers.js-specific configuration, possibly present in config.json under the key `transformers.js_config`.
 * @typedef {Object} TransformersJSConfig
 * @property {Record<import('./utils/devices.js').DeviceType, DeviceConfig>} [device_config] Device-specific configurations.
 * @property {Record<string, number>} [free_dimension_overrides] Override the free dimensions of the model.
 * See https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html#freedimensionoverrides
 * for more information.
 * @property {import('./utils/devices.js').DeviceType} [device] The default device to use for the model.
 * @property {import('./utils/dtypes.js').DataType|Record<string, import('./utils/dtypes.js').DataType>} [dtype] The default data type to use for the model.
 * @property {import('./utils/hub.js').ExternalData|Record<string, import('./utils/hub.js').ExternalData>} [use_external_data_format=false] Whether to load the model using the external data format (used for models >= 2GB in size).
 */

/**
 * Device-specific configuration options.
 * @typedef {Omit<TransformersJSConfig, "device" | "device_config">} DeviceConfig
 */
