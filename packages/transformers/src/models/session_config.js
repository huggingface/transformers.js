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
    VoxtralRealtime: 16,
};

export const MODEL_SESSION_CONFIG = {
    [MODEL_TYPES.DecoderOnly]: {
        sessions: (config, options) => ({ model: options.model_file_name ?? 'model' }),
        cache_sessions: { model: true },
        optional_configs: { generation_config: 'generation_config.json' },
    },
    [MODEL_TYPES.DecoderOnlyWithoutHead]: {
        sessions: (config, options) => ({ model: options.model_file_name ?? 'model' }),
    },
    [MODEL_TYPES.Seq2Seq]: {
        sessions: () => ({ model: 'encoder_model', decoder_model_merged: 'decoder_model_merged' }),
        cache_sessions: { decoder_model_merged: true },
        optional_configs: { generation_config: 'generation_config.json' },
    },
    [MODEL_TYPES.Vision2Seq]: {
        sessions: () => ({ model: 'encoder_model', decoder_model_merged: 'decoder_model_merged' }),
        cache_sessions: { decoder_model_merged: true },
        optional_configs: { generation_config: 'generation_config.json' },
    },
    [MODEL_TYPES.Musicgen]: {
        sessions: () => ({
            model: 'text_encoder',
            decoder_model_merged: 'decoder_model_merged',
            encodec_decode: 'encodec_decode',
        }),
        cache_sessions: { decoder_model_merged: true },
        optional_configs: { generation_config: 'generation_config.json' },
    },
    [MODEL_TYPES.EncoderDecoder]: {
        sessions: () => ({ model: 'encoder_model', decoder_model_merged: 'decoder_model_merged' }),
        cache_sessions: { decoder_model_merged: true },
    },
    [MODEL_TYPES.MaskGeneration]: {
        sessions: () => ({ model: 'vision_encoder', prompt_encoder_mask_decoder: 'prompt_encoder_mask_decoder' }),
    },
    [MODEL_TYPES.ImageTextToText]: {
        text_only_sessions: { embed_tokens: 'embed_tokens', decoder_model_merged: 'decoder_model_merged' },
        sessions: (config, options, textOnly) => {
            const s = { ...MODEL_SESSION_CONFIG[MODEL_TYPES.ImageTextToText].text_only_sessions };
            if (!textOnly) s['vision_encoder'] = 'vision_encoder';
            if (config.is_encoder_decoder) s['model'] = 'encoder_model';
            return s;
        },
        cache_sessions: { decoder_model_merged: true },
        optional_configs: { generation_config: 'generation_config.json' },
    },
    [MODEL_TYPES.AudioTextToText]: {
        text_only_sessions: { embed_tokens: 'embed_tokens', decoder_model_merged: 'decoder_model_merged' },
        sessions: (config, options, textOnly) => {
            const s = { ...MODEL_SESSION_CONFIG[MODEL_TYPES.AudioTextToText].text_only_sessions };
            if (!textOnly) s['audio_encoder'] = 'audio_encoder';
            return s;
        },
        cache_sessions: { decoder_model_merged: true },
        optional_configs: { generation_config: 'generation_config.json' },
    },
    [MODEL_TYPES.ImageAudioTextToText]: {
        text_only_sessions: { embed_tokens: 'embed_tokens', decoder_model_merged: 'decoder_model_merged' },
        sessions: (config, options, textOnly) => {
            const s = { ...MODEL_SESSION_CONFIG[MODEL_TYPES.ImageAudioTextToText].text_only_sessions };
            if (!textOnly) {
                s['audio_encoder'] = 'audio_encoder';
                s['vision_encoder'] = 'vision_encoder';
            }
            return s;
        },
        optional_configs: { generation_config: 'generation_config.json' },
    },
    [MODEL_TYPES.Phi3V]: {
        sessions: () => ({
            prepare_inputs_embeds: 'prepare_inputs_embeds',
            model: 'model',
            vision_encoder: 'vision_encoder',
        }),
        cache_sessions: { model: true },
        optional_configs: { generation_config: 'generation_config.json' },
    },
    [MODEL_TYPES.MultiModality]: {
        sessions: () => ({
            prepare_inputs_embeds: 'prepare_inputs_embeds',
            model: 'language_model',
            lm_head: 'lm_head',
            gen_head: 'gen_head',
            gen_img_embeds: 'gen_img_embeds',
            image_decode: 'image_decode',
        }),
        cache_sessions: { model: true },
        optional_configs: { generation_config: 'generation_config.json' },
    },
    [MODEL_TYPES.AutoEncoder]: {
        sessions: () => ({ encoder_model: 'encoder_model', decoder_model: 'decoder_model' }),
    },
    [MODEL_TYPES.Supertonic]: {
        sessions: () => ({
            text_encoder: 'text_encoder',
            latent_denoiser: 'latent_denoiser',
            voice_decoder: 'voice_decoder',
        }),
    },
    [MODEL_TYPES.Chatterbox]: {
        sessions: () => ({
            embed_tokens: 'embed_tokens',
            speech_encoder: 'speech_encoder',
            model: 'language_model',
            conditional_decoder: 'conditional_decoder',
        }),
        cache_sessions: { model: true },
        optional_configs: { generation_config: 'generation_config.json' },
    },
    [MODEL_TYPES.VoxtralRealtime]: {
        text_only_sessions: { embed_tokens: 'embed_tokens', decoder_model_merged: 'decoder_model_merged' },
        sessions: (config, options, textOnly) => {
            const s = { ...MODEL_SESSION_CONFIG[MODEL_TYPES.VoxtralRealtime].text_only_sessions };
            if (!textOnly) s['audio_encoder'] = 'audio_encoder';
            return s;
        },
        cache_sessions: { decoder_model_merged: true, audio_encoder: true },
        optional_configs: { generation_config: 'generation_config.json' },
    },
    default: {
        sessions: (config, options) => ({ model: options.model_file_name ?? 'model' }),
    },
};

/**
 * Returns the text-only session names for a given model type, or `null` if
 * the model type does not define a text-only session set.
 * @param {number} modelType The model type enum value.
 * @returns {Record<string, string>|null}
 */
export function getTextOnlySessions(modelType) {
    const typeConfig = MODEL_SESSION_CONFIG[modelType];
    return typeConfig?.text_only_sessions ?? null;
}

/**
 * Get the session configuration for a given model type.
 * @param {number} modelType The model type enum value.
 * @param {Object} config The model config.
 * @param {Object} [options] Loading options.
 * @returns {{ sessions: Record<string, string>, cache_sessions?: Record<string, true>, optional_configs?: Record<string, string> }}
 */
export function getSessionsConfig(modelType, config, options = {}) {
    const typeConfig = MODEL_SESSION_CONFIG[modelType] ?? MODEL_SESSION_CONFIG.default;
    return {
        sessions: typeConfig.sessions(config, options, options.textOnly ?? false),
        cache_sessions: typeConfig.cache_sessions,
        optional_configs: typeConfig.optional_configs,
    };
}
