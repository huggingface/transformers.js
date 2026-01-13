import {
    MODEL_TYPES,
    MODEL_TYPE_MAPPING,
    MODEL_NAME_TO_CLASS_MAPPING,
    MODEL_CLASS_TO_NAME_MAPPING,
    PreTrainedModel,
    registerTaskMappings,
} from './modeling_utils.js';

import * as ALL_MODEL_FILES from './index.js';

const MODEL_MAPPING_NAMES_ENCODER_ONLY = new Map([
    ['bert', 'BertModel'],
    ['neobert', 'NeoBertModel'],
    ['modernbert', 'ModernBertModel'],
    ['nomic_bert', 'NomicBertModel'],
    ['roformer', 'RoFormerModel'],
    ['electra', 'ElectraModel'],
    ['esm', 'EsmModel'],
    ['convbert', 'ConvBertModel'],
    ['camembert', 'CamembertModel'],
    ['deberta', 'DebertaModel'],
    ['deberta-v2', 'DebertaV2Model'],
    ['mpnet', 'MPNetModel'],
    ['albert', 'AlbertModel'],
    ['distilbert', 'DistilBertModel'],
    ['roberta', 'RobertaModel'],
    ['xlm', 'XLMModel'],
    ['xlm-roberta', 'XLMRobertaModel'],
    ['clap', 'ClapModel'],
    ['clip', 'CLIPModel'],
    ['clipseg', 'CLIPSegModel'],
    ['chinese_clip', 'ChineseCLIPModel'],
    ['siglip', 'SiglipModel'],
    ['jina_clip', 'JinaCLIPModel'],
    ['mobilebert', 'MobileBertModel'],
    ['squeezebert', 'SqueezeBertModel'],
    ['wav2vec2', 'Wav2Vec2Model'],
    ['wav2vec2-bert', 'Wav2Vec2BertModel'],
    ['unispeech', 'UniSpeechModel'],
    ['unispeech-sat', 'UniSpeechSatModel'],
    ['hubert', 'HubertModel'],
    ['wavlm', 'WavLMModel'],
    ['audio-spectrogram-transformer', 'ASTModel'],
    ['vits', 'VitsModel'],
    ['pyannote', 'PyAnnoteModel'],
    ['wespeaker-resnet', 'WeSpeakerResNetModel'],

    ['detr', 'DetrModel'],
    ['rt_detr', 'RTDetrModel'],
    ['rt_detr_v2', 'RTDetrV2Model'],
    ['rf_detr', 'RFDetrModel'],
    ['d_fine', 'DFineModel'],
    ['table-transformer', 'TableTransformerModel'],
    ['vit', 'ViTModel'],
    ['ijepa', 'IJepaModel'],
    ['pvt', 'PvtModel'],
    ['vit_msn', 'ViTMSNModel'],
    ['vit_mae', 'ViTMAEModel'],
    ['groupvit', 'GroupViTModel'],
    ['fastvit', 'FastViTModel'],
    ['mobilevit', 'MobileViTModel'],
    ['mobilevitv2', 'MobileViTV2Model'],
    ['owlvit', 'OwlViTModel'],
    ['owlv2', 'Owlv2Model'],
    ['beit', 'BeitModel'],
    ['deit', 'DeiTModel'],
    ['hiera', 'HieraModel'],
    ['convnext', 'ConvNextModel'],
    ['convnextv2', 'ConvNextV2Model'],
    ['dinov2', 'Dinov2Model'],
    ['dinov2_with_registers', 'Dinov2WithRegistersModel'],
    ['dinov3_vit', 'DINOv3ViTModel'],
    ['dinov3_convnext', 'DINOv3ConvNextModel'],
    ['resnet', 'ResNetModel'],
    ['swin', 'SwinModel'],
    ['swin2sr', 'Swin2SRModel'],
    ['donut-swin', 'DonutSwinModel'],
    ['yolos', 'YolosModel'],
    ['dpt', 'DPTModel'],
    ['glpn', 'GLPNModel'],

    ['hifigan', 'SpeechT5HifiGan'],
    ['efficientnet', 'EfficientNetModel'],

    ['decision_transformer', 'DecisionTransformerModel'],
    ['patchtst', 'PatchTSTForPrediction'],
    ['patchtsmixer', 'PatchTSMixerForPrediction'],

    ['mobilenet_v1', 'MobileNetV1Model'],
    ['mobilenet_v2', 'MobileNetV2Model'],
    ['mobilenet_v3', 'MobileNetV3Model'],
    ['mobilenet_v4', 'MobileNetV4Model'],

    ['maskformer', 'MaskFormerModel'],
    ['mgp-str', 'MgpstrForSceneTextRecognition'],

    ['style_text_to_speech_2', 'StyleTextToSpeech2Model'],
]);

const MODEL_MAPPING_NAMES_ENCODER_DECODER = new Map([
    ['t5', 'T5Model'],
    ['longt5', 'LongT5Model'],
    ['mt5', 'MT5Model'],
    ['bart', 'BartModel'],
    ['mbart', 'MBartModel'],
    ['marian', 'MarianModel'],
    ['whisper', 'WhisperModel'],
    ['m2m_100', 'M2M100Model'],
    ['blenderbot', 'BlenderbotModel'],
    ['blenderbot-small', 'BlenderbotSmallModel'],
]);

const MODEL_MAPPING_NAMES_AUTO_ENCODER = new Map([
    ['mimi', 'MimiModel'],
    ['dac', 'DacModel'],
    ['snac', 'SnacModel'],
]);

const MODEL_MAPPING_NAMES_DECODER_ONLY = new Map([
    ['bloom', 'BloomModel'],
    ['jais', 'JAISModel'],
    ['gpt2', 'GPT2Model'],
    ['gpt_oss', 'GptOssModel'],
    ['gptj', 'GPTJModel'],
    ['gpt_bigcode', 'GPTBigCodeModel'],
    ['gpt_neo', 'GPTNeoModel'],
    ['gpt_neox', 'GPTNeoXModel'],
    ['codegen', 'CodeGenModel'],
    ['llama', 'LlamaModel'],
    ['apertus', 'ApertusModel'],
    ['nanochat', 'NanoChatModel'],
    ['arcee', 'ArceeModel'],
    ['lfm2', 'Lfm2Model'],
    ['smollm3', 'SmolLM3Model'],
    ['exaone', 'ExaoneModel'],
    ['olmo', 'OlmoModel'],
    ['olmo2', 'Olmo2Model'],
    ['olmo3', 'Olmo3Model'],
    ['mobilellm', 'MobileLLMModel'],
    ['granite', 'GraniteModel'],
    ['granitemoehybrid', 'GraniteMoeHybridModel'],
    ['cohere', 'CohereModel'],
    ['gemma', 'GemmaModel'],
    ['gemma2', 'Gemma2Model'],
    ['vaultgemma', 'VaultGemmaModel'],
    ['gemma3_text', 'Gemma3Model'],
    ['helium', 'HeliumModel'],
    ['glm', 'GlmModel'],
    ['openelm', 'OpenELMModel'],
    ['qwen2', 'Qwen2Model'],
    ['qwen3', 'Qwen3Model'],
    ['phi', 'PhiModel'],
    ['phi3', 'Phi3Model'],
    ['mpt', 'MptModel'],
    ['opt', 'OPTModel'],
    ['mistral', 'MistralModel'],
    ['ernie4_5', 'Ernie4_5_Model'],
    ['starcoder2', 'Starcoder2Model'],
    ['falcon', 'FalconModel'],
    ['stablelm', 'StableLmModel'],
    ['modernbert-decoder', 'ModernBertDecoderModel'],
]);

export const MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES = new Map([
    ['speecht5', 'SpeechT5ForSpeechToText'],
    ['whisper', 'WhisperForConditionalGeneration'],
    ['lite-whisper', 'LiteWhisperForConditionalGeneration'],
    ['moonshine', 'MoonshineForConditionalGeneration'],
]);

const MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES = new Map([['speecht5', 'SpeechT5ForTextToSpeech']]);

const MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES = new Map([
    ['vits', 'VitsModel'],
    ['musicgen', 'MusicgenForConditionalGeneration'],
    ['supertonic', 'SupertonicForConditionalGeneration'],
]);

const MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = new Map([
    ['bert', 'BertForSequenceClassification'],
    ['neobert', 'NeoBertForSequenceClassification'],
    ['modernbert', 'ModernBertForSequenceClassification'],
    ['roformer', 'RoFormerForSequenceClassification'],
    ['electra', 'ElectraForSequenceClassification'],
    ['esm', 'EsmForSequenceClassification'],
    ['convbert', 'ConvBertForSequenceClassification'],
    ['camembert', 'CamembertForSequenceClassification'],
    ['deberta', 'DebertaForSequenceClassification'],
    ['deberta-v2', 'DebertaV2ForSequenceClassification'],
    ['mpnet', 'MPNetForSequenceClassification'],
    ['albert', 'AlbertForSequenceClassification'],
    ['distilbert', 'DistilBertForSequenceClassification'],
    ['roberta', 'RobertaForSequenceClassification'],
    ['xlm', 'XLMForSequenceClassification'],
    ['xlm-roberta', 'XLMRobertaForSequenceClassification'],
    ['bart', 'BartForSequenceClassification'],
    ['mbart', 'MBartForSequenceClassification'],
    ['mobilebert', 'MobileBertForSequenceClassification'],
    ['squeezebert', 'SqueezeBertForSequenceClassification'],
]);

const MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES = new Map([
    ['bert', 'BertForTokenClassification'],
    ['neobert', 'NeoBertForTokenClassification'],
    ['modernbert', 'ModernBertForTokenClassification'],
    ['roformer', 'RoFormerForTokenClassification'],
    ['electra', 'ElectraForTokenClassification'],
    ['esm', 'EsmForTokenClassification'],
    ['convbert', 'ConvBertForTokenClassification'],
    ['camembert', 'CamembertForTokenClassification'],
    ['deberta', 'DebertaForTokenClassification'],
    ['deberta-v2', 'DebertaV2ForTokenClassification'],
    ['mpnet', 'MPNetForTokenClassification'],
    ['distilbert', 'DistilBertForTokenClassification'],
    ['roberta', 'RobertaForTokenClassification'],
    ['xlm', 'XLMForTokenClassification'],
    ['xlm-roberta', 'XLMRobertaForTokenClassification'],
]);

export const MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES = new Map([
    ['t5', 'T5ForConditionalGeneration'],
    ['longt5', 'LongT5ForConditionalGeneration'],
    ['mt5', 'MT5ForConditionalGeneration'],
    ['bart', 'BartForConditionalGeneration'],
    ['mbart', 'MBartForConditionalGeneration'],
    ['marian', 'MarianMTModel'],
    ['m2m_100', 'M2M100ForConditionalGeneration'],
    ['blenderbot', 'BlenderbotForConditionalGeneration'],
    ['blenderbot-small', 'BlenderbotSmallForConditionalGeneration'],
]);

export const MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = new Map([
    ['bloom', 'BloomForCausalLM'],
    ['gpt2', 'GPT2LMHeadModel'],
    ['gpt_oss', 'GptOssForCausalLM'],
    ['jais', 'JAISLMHeadModel'],
    ['gptj', 'GPTJForCausalLM'],
    ['gpt_bigcode', 'GPTBigCodeForCausalLM'],
    ['gpt_neo', 'GPTNeoForCausalLM'],
    ['gpt_neox', 'GPTNeoXForCausalLM'],
    ['codegen', 'CodeGenForCausalLM'],
    ['llama', 'LlamaForCausalLM'],
    ['nanochat', 'NanoChatForCausalLM'],
    ['apertus', 'ApertusForCausalLM'],
    ['llama4_text', 'Llama4ForCausalLM'],
    ['arcee', 'ArceeForCausalLM'],
    ['lfm2', 'Lfm2ForCausalLM'],
    ['smollm3', 'SmolLM3ForCausalLM'],
    ['exaone', 'ExaoneForCausalLM'],
    ['olmo', 'OlmoForCausalLM'],
    ['olmo2', 'Olmo2ForCausalLM'],
    ['olmo3', 'Olmo3ForCausalLM'],
    ['mobilellm', 'MobileLLMForCausalLM'],
    ['granite', 'GraniteForCausalLM'],
    ['granitemoehybrid', 'GraniteMoeHybridForCausalLM'],
    ['cohere', 'CohereForCausalLM'],
    ['gemma', 'GemmaForCausalLM'],
    ['gemma2', 'Gemma2ForCausalLM'],
    ['vaultgemma', 'VaultGemmaForCausalLM'],
    ['gemma3_text', 'Gemma3ForCausalLM'],
    ['helium', 'HeliumForCausalLM'],
    ['glm', 'GlmForCausalLM'],
    ['openelm', 'OpenELMForCausalLM'],
    ['qwen2', 'Qwen2ForCausalLM'],
    ['qwen3', 'Qwen3ForCausalLM'],
    ['phi', 'PhiForCausalLM'],
    ['phi3', 'Phi3ForCausalLM'],
    ['mpt', 'MptForCausalLM'],
    ['opt', 'OPTForCausalLM'],
    ['mbart', 'MBartForCausalLM'],
    ['mistral', 'MistralForCausalLM'],
    ['ernie4_5', 'Ernie4_5_ForCausalLM'],
    ['starcoder2', 'Starcoder2ForCausalLM'],
    ['falcon', 'FalconForCausalLM'],
    ['trocr', 'TrOCRForCausalLM'],
    ['stablelm', 'StableLmForCausalLM'],
    ['modernbert-decoder', 'ModernBertDecoderForCausalLM'],

    // Also image-text-to-text
    ['phi3_v', 'Phi3VForCausalLM'],
]);

const MODEL_FOR_MULTIMODALITY_MAPPING_NAMES = new Map([['multi_modality', 'MultiModalityCausalLM']]);

const MODEL_FOR_MASKED_LM_MAPPING_NAMES = new Map([
    ['bert', 'BertForMaskedLM'],
    ['neobert', 'NeoBertForMaskedLM'],
    ['modernbert', 'ModernBertForMaskedLM'],
    ['roformer', 'RoFormerForMaskedLM'],
    ['electra', 'ElectraForMaskedLM'],
    ['esm', 'EsmForMaskedLM'],
    ['convbert', 'ConvBertForMaskedLM'],
    ['camembert', 'CamembertForMaskedLM'],
    ['deberta', 'DebertaForMaskedLM'],
    ['deberta-v2', 'DebertaV2ForMaskedLM'],
    ['mpnet', 'MPNetForMaskedLM'],
    ['albert', 'AlbertForMaskedLM'],
    ['distilbert', 'DistilBertForMaskedLM'],
    ['roberta', 'RobertaForMaskedLM'],
    ['xlm', 'XLMWithLMHeadModel'],
    ['xlm-roberta', 'XLMRobertaForMaskedLM'],
    ['mobilebert', 'MobileBertForMaskedLM'],
    ['squeezebert', 'SqueezeBertForMaskedLM'],
]);

const MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = new Map([
    ['bert', 'BertForQuestionAnswering'],
    ['neobert', 'NeoBertForQuestionAnswering'],
    ['roformer', 'RoFormerForQuestionAnswering'],
    ['electra', 'ElectraForQuestionAnswering'],
    ['convbert', 'ConvBertForQuestionAnswering'],
    ['camembert', 'CamembertForQuestionAnswering'],
    ['deberta', 'DebertaForQuestionAnswering'],
    ['deberta-v2', 'DebertaV2ForQuestionAnswering'],
    ['mpnet', 'MPNetForQuestionAnswering'],
    ['albert', 'AlbertForQuestionAnswering'],
    ['distilbert', 'DistilBertForQuestionAnswering'],
    ['roberta', 'RobertaForQuestionAnswering'],
    ['xlm', 'XLMForQuestionAnswering'],
    ['xlm-roberta', 'XLMRobertaForQuestionAnswering'],
    ['mobilebert', 'MobileBertForQuestionAnswering'],
    ['squeezebert', 'SqueezeBertForQuestionAnswering'],
]);

export const MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = new Map([
    ['vision-encoder-decoder', 'VisionEncoderDecoderModel'],
    ['idefics3', 'Idefics3ForConditionalGeneration'],
    ['smolvlm', 'SmolVLMForConditionalGeneration'],
]);

const MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES = new Map([
    ['llava', 'LlavaForConditionalGeneration'],
    ['llava_onevision', 'LlavaOnevisionForConditionalGeneration'],
    ['moondream1', 'Moondream1ForConditionalGeneration'],
    ['florence2', 'Florence2ForConditionalGeneration'],
    ['qwen2-vl', 'Qwen2VLForConditionalGeneration'],
    ['idefics3', 'Idefics3ForConditionalGeneration'],
    ['smolvlm', 'SmolVLMForConditionalGeneration'],
    ['paligemma', 'PaliGemmaForConditionalGeneration'],
    ['llava_qwen2', 'LlavaQwen2ForCausalLM'],
    ['gemma3n', 'Gemma3nForConditionalGeneration'],
]);

const MODEL_FOR_AUDIO_TEXT_TO_TEXT_MAPPING_NAMES = new Map([
    ['ultravox', 'UltravoxModel'],
    ['voxtral', 'VoxtralForConditionalGeneration'],
]);

const MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES = new Map([
    ['vision-encoder-decoder', 'VisionEncoderDecoderModel'],
]);

const MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES = new Map([
    ['vit', 'ViTForImageClassification'],
    ['ijepa', 'IJepaForImageClassification'],
    ['pvt', 'PvtForImageClassification'],
    ['vit_msn', 'ViTMSNForImageClassification'],
    ['fastvit', 'FastViTForImageClassification'],
    ['mobilevit', 'MobileViTForImageClassification'],
    ['mobilevitv2', 'MobileViTV2ForImageClassification'],
    ['beit', 'BeitForImageClassification'],
    ['deit', 'DeiTForImageClassification'],
    ['hiera', 'HieraForImageClassification'],
    ['convnext', 'ConvNextForImageClassification'],
    ['convnextv2', 'ConvNextV2ForImageClassification'],
    ['dinov2', 'Dinov2ForImageClassification'],
    ['dinov2_with_registers', 'Dinov2WithRegistersForImageClassification'],
    ['resnet', 'ResNetForImageClassification'],
    ['swin', 'SwinForImageClassification'],
    ['segformer', 'SegformerForImageClassification'],
    ['efficientnet', 'EfficientNetForImageClassification'],
    ['mobilenet_v1', 'MobileNetV1ForImageClassification'],
    ['mobilenet_v2', 'MobileNetV2ForImageClassification'],
    ['mobilenet_v3', 'MobileNetV3ForImageClassification'],
    ['mobilenet_v4', 'MobileNetV4ForImageClassification'],
]);

const MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES = new Map([
    ['detr', 'DetrForObjectDetection'],
    ['rt_detr', 'RTDetrForObjectDetection'],
    ['rt_detr_v2', 'RTDetrV2ForObjectDetection'],
    ['rf_detr', 'RFDetrForObjectDetection'],
    ['d_fine', 'DFineForObjectDetection'],
    ['table-transformer', 'TableTransformerForObjectDetection'],
    ['yolos', 'YolosForObjectDetection'],
]);

const MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES = new Map([
    ['owlvit', 'OwlViTForObjectDetection'],
    ['owlv2', 'Owlv2ForObjectDetection'],
    ['grounding-dino', 'GroundingDinoForObjectDetection'],
]);

const MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES = new Map([
    // TODO: Do not add new models here
    ['detr', 'DetrForSegmentation'],
    ['clipseg', 'CLIPSegForImageSegmentation'],
]);

const MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES = new Map([
    ['segformer', 'SegformerForSemanticSegmentation'],
    ['sapiens', 'SapiensForSemanticSegmentation'],

    ['swin', 'SwinForSemanticSegmentation'],
    ['mobilenet_v1', 'MobileNetV1ForSemanticSegmentation'],
    ['mobilenet_v2', 'MobileNetV2ForSemanticSegmentation'],
    ['mobilenet_v3', 'MobileNetV3ForSemanticSegmentation'],
    ['mobilenet_v4', 'MobileNetV4ForSemanticSegmentation'],
]);

const MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES = new Map([
    ['detr', 'DetrForSegmentation'],
    ['maskformer', 'MaskFormerForInstanceSegmentation'],
]);

const MODEL_FOR_MASK_GENERATION_MAPPING_NAMES = new Map([
    ['sam', 'SamModel'],
    ['sam2', 'Sam2Model'],
    ['edgetam', 'EdgeTamModel'],
    ['sam3_tracker', 'Sam3TrackerModel'],
]);

const MODEL_FOR_CTC_MAPPING_NAMES = new Map([
    ['wav2vec2', 'Wav2Vec2ForCTC'],
    ['wav2vec2-bert', 'Wav2Vec2BertForCTC'],
    ['unispeech', 'UniSpeechForCTC'],
    ['unispeech-sat', 'UniSpeechSatForCTC'],
    ['wavlm', 'WavLMForCTC'],
    ['hubert', 'HubertForCTC'],
    ['parakeet_ctc', 'ParakeetForCTC'],
]);

const MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES = new Map([
    ['wav2vec2', 'Wav2Vec2ForSequenceClassification'],
    ['wav2vec2-bert', 'Wav2Vec2BertForSequenceClassification'],
    ['unispeech', 'UniSpeechForSequenceClassification'],
    ['unispeech-sat', 'UniSpeechSatForSequenceClassification'],
    ['wavlm', 'WavLMForSequenceClassification'],
    ['hubert', 'HubertForSequenceClassification'],
    ['audio-spectrogram-transformer', 'ASTForAudioClassification'],
]);

const MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES = new Map([['wavlm', 'WavLMForXVector']]);

const MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES = new Map([
    ['unispeech-sat', 'UniSpeechSatForAudioFrameClassification'],
    ['wavlm', 'WavLMForAudioFrameClassification'],
    ['wav2vec2', 'Wav2Vec2ForAudioFrameClassification'],
    ['pyannote', 'PyAnnoteForAudioFrameClassification'],
]);

const MODEL_FOR_IMAGE_MATTING_MAPPING_NAMES = new Map([['vitmatte', 'VitMatteForImageMatting']]);

const MODEL_FOR_TIME_SERIES_PREDICTION_MAPPING_NAMES = new Map([
    ['patchtst', 'PatchTSTForPrediction'],
    ['patchtsmixer', 'PatchTSMixerForPrediction'],
]);

const MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES = new Map([['swin2sr', 'Swin2SRForImageSuperResolution']]);

const MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES = new Map([
    ['dpt', 'DPTForDepthEstimation'],
    ['depth_anything', 'DepthAnythingForDepthEstimation'],
    ['glpn', 'GLPNForDepthEstimation'],
    ['sapiens', 'SapiensForDepthEstimation'],
    ['depth_pro', 'DepthProForDepthEstimation'],
    ['metric3d', 'Metric3DForDepthEstimation'],
    ['metric3dv2', 'Metric3Dv2ForDepthEstimation'],
]);

const MODEL_FOR_NORMAL_ESTIMATION_MAPPING_NAMES = new Map([['sapiens', 'SapiensForNormalEstimation']]);

const MODEL_FOR_POSE_ESTIMATION_MAPPING_NAMES = new Map([['vitpose', 'VitPoseForPoseEstimation']]);

// NOTE: This is custom to Transformers.js, and is necessary because certain models
// (e.g., CLIP) are split into vision and text components
const MODEL_FOR_IMAGE_FEATURE_EXTRACTION_MAPPING_NAMES = new Map([
    ['clip', 'CLIPVisionModelWithProjection'],
    ['siglip', 'SiglipVisionModel'],
    ['jina_clip', 'JinaCLIPVisionModel'],
]);

export const MODEL_CLASS_TYPE_MAPPING = [
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
    for (const name of mappings.values()) {
        MODEL_TYPE_MAPPING.set(name, type);
        const model = ALL_MODEL_FILES[name];
        MODEL_CLASS_TO_NAME_MAPPING.set(model, name);
        MODEL_NAME_TO_CLASS_MAPPING.set(name, model);
    }
}

const CUSTOM_MAPPING = [
    // OVERRIDE:
    // TODO: Refactor to allow class to specify model
    ['MusicgenForConditionalGeneration', ALL_MODEL_FILES.MusicgenForConditionalGeneration, MODEL_TYPES.Musicgen],
    ['Phi3VForCausalLM', ALL_MODEL_FILES.Phi3VForCausalLM, MODEL_TYPES.Phi3V],

    ['CLIPTextModelWithProjection', ALL_MODEL_FILES.CLIPTextModelWithProjection, MODEL_TYPES.EncoderOnly],
    ['SiglipTextModel', ALL_MODEL_FILES.SiglipTextModel, MODEL_TYPES.EncoderOnly],
    ['JinaCLIPTextModel', ALL_MODEL_FILES.JinaCLIPTextModel, MODEL_TYPES.EncoderOnly],
    ['ClapTextModelWithProjection', ALL_MODEL_FILES.ClapTextModelWithProjection, MODEL_TYPES.EncoderOnly],
    ['ClapAudioModelWithProjection', ALL_MODEL_FILES.ClapAudioModelWithProjection, MODEL_TYPES.EncoderOnly],

    ['DacEncoderModel', ALL_MODEL_FILES.DacEncoderModel, MODEL_TYPES.EncoderOnly],
    ['DacDecoderModel', ALL_MODEL_FILES.DacDecoderModel, MODEL_TYPES.EncoderOnly],
    ['MimiEncoderModel', ALL_MODEL_FILES.MimiEncoderModel, MODEL_TYPES.EncoderOnly],
    ['MimiDecoderModel', ALL_MODEL_FILES.MimiDecoderModel, MODEL_TYPES.EncoderOnly],
    ['SnacEncoderModel', ALL_MODEL_FILES.SnacEncoderModel, MODEL_TYPES.EncoderOnly],
    ['SnacDecoderModel', ALL_MODEL_FILES.SnacDecoderModel, MODEL_TYPES.EncoderOnly],

    [
        'Gemma3nForConditionalGeneration',
        ALL_MODEL_FILES.Gemma3nForConditionalGeneration,
        MODEL_TYPES.ImageAudioTextToText,
    ],
    ['SupertonicForConditionalGeneration', ALL_MODEL_FILES.SupertonicForConditionalGeneration, MODEL_TYPES.Supertonic],
    ['ChatterboxModel', ALL_MODEL_FILES.ChatterboxModel, MODEL_TYPES.Chatterbox],
];
for (const [name, model, type] of CUSTOM_MAPPING) {
    MODEL_TYPE_MAPPING.set(name, type);
    MODEL_CLASS_TO_NAME_MAPPING.set(model, name);
    MODEL_NAME_TO_CLASS_MAPPING.set(name, model);
}

export const CUSTOM_ARCHITECTURES = new Map([
    ['modnet', MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES],
    ['birefnet', MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES],
    ['isnet', MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES],
    ['ben', MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES],
]);
for (const [name, mapping] of CUSTOM_ARCHITECTURES.entries()) {
    mapping.set(name, 'PreTrainedModel');
    MODEL_TYPE_MAPPING.set(name, MODEL_TYPES.EncoderOnly);
    MODEL_NAME_TO_CLASS_MAPPING.set(name, PreTrainedModel);
}

// Default mappings
MODEL_TYPE_MAPPING.set('PreTrainedModel', MODEL_TYPES.EncoderOnly);
MODEL_CLASS_TO_NAME_MAPPING.set(PreTrainedModel, 'PreTrainedModel');

export const MODEL_MAPPINGS = {
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES,
    MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES,
    MODEL_FOR_MASKED_LM_MAPPING_NAMES,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES,
    MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES,
    MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES,
    MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES,
    MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES,
    MODEL_FOR_MASK_GENERATION_MAPPING_NAMES,
    MODEL_FOR_CTC_MAPPING_NAMES,
    MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES,
    MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_IMAGE_MATTING_MAPPING_NAMES,
    MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES,
    MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES,
    MODEL_FOR_NORMAL_ESTIMATION_MAPPING_NAMES,
    MODEL_FOR_POSE_ESTIMATION_MAPPING_NAMES,
    MODEL_FOR_IMAGE_FEATURE_EXTRACTION_MAPPING_NAMES,
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
    MODEL_FOR_AUDIO_TEXT_TO_TEXT_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES,
};
registerTaskMappings(MODEL_MAPPINGS);
