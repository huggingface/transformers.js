const MODEL_TYPES = {
    EncoderOnly: 'EncoderOnly',
    EncoderDecoder: 'EncoderDecoder',
    Seq2Seq: 'Seq2Seq',
    Vision2Seq: 'Vision2Seq',
    DecoderOnly: 'DecoderOnly',
    DecoderOnlyWithoutHead: 'DecoderOnlyWithoutHead',
    MaskGeneration: 'MaskGeneration',
    ImageTextToText: 'ImageTextToText',
    Musicgen: 'Musicgen',
    MultiModality: 'MultiModality',
    Phi3V: 'Phi3V',
    AudioTextToText: 'AudioTextToText',
    AutoEncoder: 'AutoEncoder',
    ImageAudioTextToText: 'ImageAudioTextToText',
    Supertonic: 'Supertonic',
    Chatterbox: 'Chatterbox',
    VoxtralRealtime: 'VoxtralRealtime',
} as const;

type SessionConfig = {
    sessions(config: any, options: any, textOnly?: boolean): Record<string, string>;
    cache_sessions?: Record<string, true>;
    optional_configs?: Record<string, string>;
    text_only_sessions?: Record<string, string>;
};

const MODEL_SESSION_CONFIG: Record<string | number, SessionConfig> = {
    [MODEL_TYPES.DecoderOnly]: {
        sessions: (_config, options) => ({ model: options.model_file_name ?? 'model' }),
        cache_sessions: { model: true },
        optional_configs: { generation_config: 'generation_config.json' },
    },
    [MODEL_TYPES.DecoderOnlyWithoutHead]: {
        sessions: (_config, options) => ({ model: options.model_file_name ?? 'model' }),
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
        sessions: (config, _options, textOnly) => {
            const sessions = { ...MODEL_SESSION_CONFIG[MODEL_TYPES.ImageTextToText].text_only_sessions };
            if (!textOnly) sessions.vision_encoder = 'vision_encoder';
            if (config.is_encoder_decoder) sessions.model = 'encoder_model';
            return sessions;
        },
        cache_sessions: { decoder_model_merged: true },
        optional_configs: { generation_config: 'generation_config.json' },
    },
    [MODEL_TYPES.AudioTextToText]: {
        text_only_sessions: { embed_tokens: 'embed_tokens', decoder_model_merged: 'decoder_model_merged' },
        sessions: (_config, _options, textOnly) => ({
            ...MODEL_SESSION_CONFIG[MODEL_TYPES.AudioTextToText].text_only_sessions,
            ...(textOnly ? {} : { audio_encoder: 'audio_encoder' }),
        }),
        cache_sessions: { decoder_model_merged: true },
        optional_configs: { generation_config: 'generation_config.json' },
    },
    [MODEL_TYPES.ImageAudioTextToText]: {
        text_only_sessions: { embed_tokens: 'embed_tokens', decoder_model_merged: 'decoder_model_merged' },
        sessions: (_config, _options, textOnly) => ({
            ...MODEL_SESSION_CONFIG[MODEL_TYPES.ImageAudioTextToText].text_only_sessions,
            ...(textOnly ? {} : { audio_encoder: 'audio_encoder', vision_encoder: 'vision_encoder' }),
        }),
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
        sessions: () => ({ text_encoder: 'text_encoder', latent_denoiser: 'latent_denoiser', voice_decoder: 'voice_decoder' }),
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
        sessions: (_config, _options, textOnly) => ({
            ...MODEL_SESSION_CONFIG[MODEL_TYPES.VoxtralRealtime].text_only_sessions,
            ...(textOnly ? {} : { audio_encoder: 'audio_encoder' }),
        }),
        cache_sessions: { decoder_model_merged: true, audio_encoder: true },
        optional_configs: { generation_config: 'generation_config.json' },
    },
    default: {
        sessions: (_config, options) => ({ model: options.model_file_name ?? 'model' }),
    },
};

export function getSessionsConfig(modelType: string, config: any, options: any = {}) {
    const typeConfig = MODEL_SESSION_CONFIG[modelType] ?? MODEL_SESSION_CONFIG.default;
    return {
        sessions: typeConfig.sessions(config, options, options.textOnly ?? false),
        cache_sessions: typeConfig.cache_sessions,
        optional_configs: typeConfig.optional_configs,
    };
}

export function getTextOnlySessions(modelType: string): Record<string, string> | null {
    return MODEL_SESSION_CONFIG[modelType]?.text_only_sessions ?? null;
}
