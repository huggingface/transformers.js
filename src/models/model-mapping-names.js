/**
 * @file Model mapping names without class references to avoid circular dependencies.
 * These mappings are used by PreTrainedModel for model validation.
 */

export const MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES = new Map([
    ['speecht5', ['SpeechT5ForSpeechToText']],
    ['whisper', ['WhisperForConditionalGeneration']],
    ['lite-whisper', ['LiteWhisperForConditionalGeneration']],
    ['moonshine', ['MoonshineForConditionalGeneration']],
]);

export const MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES = new Map([
    ['t5', ['T5ForConditionalGeneration']],
    ['longt5', ['LongT5ForConditionalGeneration']],
    ['mt5', ['MT5ForConditionalGeneration']],
    ['bart', ['BartForConditionalGeneration']],
    ['mbart', ['MBartForConditionalGeneration']],
    ['marian', ['MarianMTModel']],
    ['m2m_100', ['M2M100ForConditionalGeneration']],
    ['blenderbot', ['BlenderbotForConditionalGeneration']],
    ['blenderbot-small', ['BlenderbotSmallForConditionalGeneration']],
]);

export const MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = new Map([
    ['bloom', ['BloomForCausalLM']],
    ['gpt2', ['GPT2LMHeadModel']],
    ['gpt_oss', ['GptOssForCausalLM']],
    ['jais', ['JAISLMHeadModel']],
    ['gptj', ['GPTJForCausalLM']],
    ['gpt_bigcode', ['GPTBigCodeForCausalLM']],
    ['gpt_neo', ['GPTNeoForCausalLM']],
    ['gpt_neox', ['GPTNeoXForCausalLM']],
    ['codegen', ['CodeGenForCausalLM']],
    ['llama', ['LlamaForCausalLM']],
    ['nanochat', ['NanoChatForCausalLM']],
    ['apertus', ['ApertusForCausalLM']],
    ['llama4_text', ['Llama4ForCausalLM']],
    ['arcee', ['ArceeForCausalLM']],
    ['lfm2', ['Lfm2ForCausalLM']],
    ['smollm3', ['SmolLM3ForCausalLM']],
    ['exaone', ['ExaoneForCausalLM']],
    ['olmo', ['OlmoForCausalLM']],
    ['olmo2', ['Olmo2ForCausalLM']],
    ['olmo3', ['Olmo3ForCausalLM']],
    ['mobilellm', ['MobileLLMForCausalLM']],
    ['granite', ['GraniteForCausalLM']],
    ['granitemoehybrid', ['GraniteMoeHybridForCausalLM']],
    ['cohere', ['CohereForCausalLM']],
    ['gemma', ['GemmaForCausalLM']],
    ['gemma2', ['Gemma2ForCausalLM']],
    ['vaultgemma', ['VaultGemmaForCausalLM']],
    ['gemma3_text', ['Gemma3ForCausalLM']],
    ['helium', ['HeliumForCausalLM']],
    ['glm', ['GlmForCausalLM']],
    ['openelm', ['OpenELMForCausalLM']],
    ['qwen2', ['Qwen2ForCausalLM']],
    ['qwen3', ['Qwen3ForCausalLM']],
    ['phi', ['PhiForCausalLM']],
    ['phi3', ['Phi3ForCausalLM']],
    ['mpt', ['MptForCausalLM']],
    ['opt', ['OPTForCausalLM']],
    ['mbart', ['MBartForCausalLM']],
    ['mistral', ['MistralForCausalLM']],
    ['ernie4_5', ['Ernie4_5_ForCausalLM']],
    ['starcoder2', ['Starcoder2ForCausalLM']],
    ['falcon', ['FalconForCausalLM']],
    ['trocr', ['TrOCRForCausalLM']],
    ['stablelm', ['StableLmForCausalLM']],
    ['modernbert-decoder', ['ModernBertDecoderForCausalLM']],
    // Also image-text-to-text
    ['phi3_v', ['Phi3VForCausalLM']],
]);

export const MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = new Map([
    ['vision-encoder-decoder', ['VisionEncoderDecoderModel']],
    ['idefics3', ['Idefics3ForConditionalGeneration']],
    ['smolvlm', ['SmolVLMForConditionalGeneration']],
]);
