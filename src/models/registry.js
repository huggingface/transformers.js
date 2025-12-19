// Import the Maps and MODEL_TYPES from pre-trained-model.js
import {
    MODEL_TYPES,
    MODEL_TYPE_MAPPING,
    MODEL_NAME_TO_CLASS_MAPPING,
    MODEL_CLASS_TO_NAME_MAPPING,
    PreTrainedModel,
} from './pre-trained-model.js';

// import all classes
import { ASTModel, ASTForAudioClassification } from './pre-trained-models/ASTPreTrainedModel.js';
import {
    AlbertModel,
    AlbertForSequenceClassification,
    AlbertForQuestionAnswering,
    AlbertForMaskedLM,
} from './pre-trained-models/AlbertPreTrainedModel.js';
import { ApertusModel, ApertusForCausalLM } from './pre-trained-models/ApertusPreTrainedModel.js';
import { ArceeModel, ArceeForCausalLM } from './pre-trained-models/ArceePreTrainedModel.js';
import {
    BartModel,
    BartForConditionalGeneration,
    BartForSequenceClassification,
} from './pre-trained-models/BartPretrainedModel.js';
import { BeitModel, BeitForImageClassification } from './pre-trained-models/BeitPreTrainedModel.js';
import {
    BertModel,
    BertForMaskedLM,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertForQuestionAnswering,
} from './pre-trained-models/BertPreTrainedModel.js';
import { BlenderbotModel, BlenderbotForConditionalGeneration } from './pre-trained-models/BlenderbotPreTrainedModel.js';
import {
    BlenderbotSmallModel,
    BlenderbotSmallForConditionalGeneration,
} from './pre-trained-models/BlenderbotSmallPreTrainedModel.js';
import { BloomModel, BloomForCausalLM } from './pre-trained-models/BloomPreTrainedModel.js';
import {
    CLIPModel,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
} from './pre-trained-models/CLIPPreTrainedModel.js';
import { CLIPSegModel, CLIPSegForImageSegmentation } from './pre-trained-models/CLIPSegPreTrainedModel.js';
import {
    CamembertModel,
    CamembertForMaskedLM,
    CamembertForSequenceClassification,
    CamembertForTokenClassification,
    CamembertForQuestionAnswering,
} from './pre-trained-models/CamembertPreTrainedModel.js';
import { ChatterboxModel } from './pre-trained-models/ChatterboxPreTrainedModel.js';
import { ChineseCLIPModel } from './pre-trained-models/ChineseCLIPPreTrainedModel.js';
import {
    ClapModel,
    ClapTextModelWithProjection,
    ClapAudioModelWithProjection,
} from './pre-trained-models/ClapPreTrainedModel.js';
import { CodeGenModel, CodeGenForCausalLM } from './pre-trained-models/CodeGenPreTrainedModel.js';
import { CohereModel, CohereForCausalLM } from './pre-trained-models/CoherePreTrainedModel.js';
import {
    ConvBertModel,
    ConvBertForMaskedLM,
    ConvBertForSequenceClassification,
    ConvBertForTokenClassification,
    ConvBertForQuestionAnswering,
} from './pre-trained-models/ConvBertPreTrainedModel.js';
import { ConvNextModel, ConvNextForImageClassification } from './pre-trained-models/ConvNextPreTrainedModel.js';
import { ConvNextV2Model, ConvNextV2ForImageClassification } from './pre-trained-models/ConvNextV2PreTrainedModel.js';
import { DFineModel, DFineForObjectDetection } from './pre-trained-models/DFinePreTrainedModel.js';
import { DINOv3ConvNextModel } from './pre-trained-models/DINOv3ConvNextPreTrainedModel.js';
import { DINOv3ViTModel } from './pre-trained-models/DINOv3ViTPreTrainedModel.js';
import { DPTModel, DPTForDepthEstimation } from './pre-trained-models/DPTPreTrainedModel.js';
import { DacModel, DacEncoderModel, DacDecoderModel } from './pre-trained-models/DacPreTrainedModel.js';
import {
    DebertaModel,
    DebertaForMaskedLM,
    DebertaForSequenceClassification,
    DebertaForTokenClassification,
    DebertaForQuestionAnswering,
} from './pre-trained-models/DebertaPreTrainedModel.js';
import {
    DebertaV2Model,
    DebertaV2ForMaskedLM,
    DebertaV2ForSequenceClassification,
    DebertaV2ForTokenClassification,
    DebertaV2ForQuestionAnswering,
} from './pre-trained-models/DebertaV2PreTrainedModel.js';
import { DecisionTransformerModel } from './pre-trained-models/DecisionTransformerPreTrainedModel.js';
import { DeiTModel, DeiTForImageClassification } from './pre-trained-models/DeiTPreTrainedModel.js';
import { DepthAnythingForDepthEstimation } from './pre-trained-models/DepthAnythingPreTrainedModel.js';
import { DepthProForDepthEstimation } from './pre-trained-models/DepthProPreTrainedModel.js';
import { DetrModel, DetrForObjectDetection, DetrForSegmentation } from './pre-trained-models/DetrPreTrainedModel.js';
import { Dinov2Model, Dinov2ForImageClassification } from './pre-trained-models/Dinov2PreTrainedModel.js';
import {
    Dinov2WithRegistersModel,
    Dinov2WithRegistersForImageClassification,
} from './pre-trained-models/Dinov2WithRegistersPreTrainedModel.js';
import {
    DistilBertModel,
    DistilBertForSequenceClassification,
    DistilBertForTokenClassification,
    DistilBertForQuestionAnswering,
    DistilBertForMaskedLM,
} from './pre-trained-models/DistilBertPreTrainedModel.js';
import { DonutSwinModel } from './pre-trained-models/DonutSwinPreTrainedModel.js';
import {
    EfficientNetModel,
    EfficientNetForImageClassification,
} from './pre-trained-models/EfficientNetPreTrainedModel.js';
import {
    ElectraModel,
    ElectraForMaskedLM,
    ElectraForSequenceClassification,
    ElectraForTokenClassification,
    ElectraForQuestionAnswering,
} from './pre-trained-models/ElectraPreTrainedModel.js';
import { Ernie4_5_Model, Ernie4_5_ForCausalLM } from './pre-trained-models/Ernie4_5_PretrainedModel.js';
import {
    EsmModel,
    EsmForMaskedLM,
    EsmForSequenceClassification,
    EsmForTokenClassification,
} from './pre-trained-models/EsmPreTrainedModel.js';
import { ExaoneModel, ExaoneForCausalLM } from './pre-trained-models/ExaonePreTrainedModel.js';
import { FalconModel, FalconForCausalLM } from './pre-trained-models/FalconPreTrainedModel.js';
import { FastViTModel, FastViTForImageClassification } from './pre-trained-models/FastViTPreTrainedModel.js';
import { Florence2ForConditionalGeneration } from './pre-trained-models/Florence2PreTrainedModel.js';
import { GLPNModel, GLPNForDepthEstimation } from './pre-trained-models/GLPNPreTrainedModel.js';
import { GPT2Model, GPT2LMHeadModel } from './pre-trained-models/GPT2PreTrainedModel.js';
import { GPTBigCodeModel, GPTBigCodeForCausalLM } from './pre-trained-models/GPTBigCodePreTrainedModel.js';
import { GPTJModel, GPTJForCausalLM } from './pre-trained-models/GPTJPreTrainedModel.js';
import { GPTNeoModel, GPTNeoForCausalLM } from './pre-trained-models/GPTNeoPreTrainedModel.js';
import { GPTNeoXModel, GPTNeoXForCausalLM } from './pre-trained-models/GPTNeoXPreTrainedModel.js';
import { Gemma2Model, Gemma2ForCausalLM } from './pre-trained-models/Gemma2PreTrainedModel.js';
import { Gemma3Model, Gemma3ForCausalLM } from './pre-trained-models/Gemma3PreTrainedModel.js';
import { Gemma3nForConditionalGeneration } from './pre-trained-models/Gemma3nPreTrainedModel.js';
import { GemmaModel, GemmaForCausalLM } from './pre-trained-models/GemmaPreTrainedModel.js';
import { GlmModel, GlmForCausalLM } from './pre-trained-models/GlmPreTrainedModel.js';
import { GptOssModel, GptOssForCausalLM } from './pre-trained-models/GptOssPreTrainedModel.js';
import {
    GraniteMoeHybridModel,
    GraniteMoeHybridForCausalLM,
} from './pre-trained-models/GraniteMoeHybridPreTrainedModel.js';
import { GraniteModel, GraniteForCausalLM } from './pre-trained-models/GranitePreTrainedModel.js';
import { GroundingDinoForObjectDetection } from './pre-trained-models/GroundingDinoPreTrainedModel.js';
import { GroupViTModel } from './pre-trained-models/GroupViTPreTrainedModel.js';
import { HeliumModel, HeliumForCausalLM } from './pre-trained-models/HeliumPreTrainedModel.js';
import { HieraModel, HieraForImageClassification } from './pre-trained-models/HieraPreTrainedModel.js';
import {
    HubertModel,
    HubertForCTC,
    HubertForSequenceClassification,
} from './pre-trained-models/HubertPreTrainedModel.js';
import { IJepaModel, IJepaForImageClassification } from './pre-trained-models/IJepaPreTrainedModel.js';
import {
    Idefics3ForConditionalGeneration,
    SmolVLMForConditionalGeneration,
} from './pre-trained-models/Idefics3PreTrainedModel.js';
import { JAISModel, JAISLMHeadModel } from './pre-trained-models/JAISPreTrainedModel.js';
import { JinaCLIPModel, JinaCLIPTextModel, JinaCLIPVisionModel } from './pre-trained-models/JinaCLIPPreTrainedModel.js';
import { Lfm2Model, Lfm2ForCausalLM } from './pre-trained-models/Lfm2PreTrainedModel.js';
import { Llama4ForCausalLM } from './pre-trained-models/Llama4PreTrainedModel.js';
import { LlamaModel, LlamaForCausalLM } from './pre-trained-models/LlamaPreTrainedModel.js';
import {
    LlavaForConditionalGeneration,
    LlavaOnevisionForConditionalGeneration,
    Moondream1ForConditionalGeneration,
    LlavaQwen2ForCausalLM,
} from './pre-trained-models/LlavaPreTrainedModel.js';
import { LongT5Model, LongT5ForConditionalGeneration } from './pre-trained-models/LongT5PreTrainedModel.js';
import { M2M100Model, M2M100ForConditionalGeneration } from './pre-trained-models/M2M100PreTrainedModel.js';
import {
    MBartModel,
    MBartForConditionalGeneration,
    MBartForSequenceClassification,
    MBartForCausalLM,
} from './pre-trained-models/MBartPreTrainedModel.js';
import {
    MPNetModel,
    MPNetForMaskedLM,
    MPNetForSequenceClassification,
    MPNetForTokenClassification,
    MPNetForQuestionAnswering,
} from './pre-trained-models/MPNetPreTrainedModel.js';
import { MT5Model, MT5ForConditionalGeneration } from './pre-trained-models/MT5PreTrainedModel.js';
import { MarianModel, MarianMTModel } from './pre-trained-models/MarianPreTrainedModel.js';
import { MaskFormerModel, MaskFormerForInstanceSegmentation } from './pre-trained-models/MaskFormerPreTrainedModel.js';
import { Metric3DForDepthEstimation } from './pre-trained-models/Metric3DPreTrainedModel.js';
import { Metric3Dv2ForDepthEstimation } from './pre-trained-models/Metric3Dv2PreTrainedModel.js';
import { MgpstrForSceneTextRecognition } from './pre-trained-models/MgpstrPreTrainedModel.js';
import { MimiModel, MimiEncoderModel, MimiDecoderModel } from './pre-trained-models/MimiPreTrainedModel.js';
import { MistralModel, MistralForCausalLM } from './pre-trained-models/MistralPreTrainedModel.js';
import {
    MobileBertModel,
    MobileBertForMaskedLM,
    MobileBertForSequenceClassification,
    MobileBertForQuestionAnswering,
} from './pre-trained-models/MobileBertPreTrainedModel.js';
import { MobileLLMModel, MobileLLMForCausalLM } from './pre-trained-models/MobileLLMPreTrainedModel.js';
import {
    MobileNetV1Model,
    MobileNetV1ForImageClassification,
    MobileNetV1ForSemanticSegmentation,
} from './pre-trained-models/MobileNetV1PreTrainedModel.js';
import {
    MobileNetV2Model,
    MobileNetV2ForImageClassification,
    MobileNetV2ForSemanticSegmentation,
} from './pre-trained-models/MobileNetV2PreTrainedModel.js';
import {
    MobileNetV3Model,
    MobileNetV3ForImageClassification,
    MobileNetV3ForSemanticSegmentation,
} from './pre-trained-models/MobileNetV3PreTrainedModel.js';
import {
    MobileNetV4Model,
    MobileNetV4ForImageClassification,
    MobileNetV4ForSemanticSegmentation,
} from './pre-trained-models/MobileNetV4PreTrainedModel.js';
import { MobileViTModel, MobileViTForImageClassification } from './pre-trained-models/MobileViTPreTrainedModel.js';
import {
    MobileViTV2Model,
    MobileViTV2ForImageClassification,
} from './pre-trained-models/MobileViTV2PreTrainedModel.js';
import {
    ModernBertDecoderModel,
    ModernBertDecoderForCausalLM,
} from './pre-trained-models/ModernBertDecoderPreTrainedModel.js';
import {
    ModernBertModel,
    ModernBertForMaskedLM,
    ModernBertForSequenceClassification,
    ModernBertForTokenClassification,
} from './pre-trained-models/ModernBertPreTrainedModel.js';
import { MoonshineForConditionalGeneration } from './pre-trained-models/MoonshinePreTrainedModel.js';
import { MptModel, MptForCausalLM } from './pre-trained-models/MptPreTrainedModel.js';
import { MultiModalityCausalLM } from './pre-trained-models/MultiModalityPreTrainedModel.js';
import { MusicgenForConditionalGeneration } from './pre-trained-models/MusicgenPreTrainedModel.js';
import { NanoChatModel, NanoChatForCausalLM } from './pre-trained-models/NanoChatPreTrainedModel.js';
import {
    NeoBertModel,
    NeoBertForMaskedLM,
    NeoBertForSequenceClassification,
    NeoBertForTokenClassification,
    NeoBertForQuestionAnswering,
} from './pre-trained-models/NeoBertPreTrainedModel.js';
import { NomicBertModel } from './pre-trained-models/NomicBertPreTrainedModel.js';
import { OPTModel, OPTForCausalLM } from './pre-trained-models/OPTPreTrainedModel.js';
import { Olmo2Model, Olmo2ForCausalLM } from './pre-trained-models/Olmo2PreTrainedModel.js';
import { Olmo3Model, Olmo3ForCausalLM } from './pre-trained-models/Olmo3PreTrainedModel.js';
import { OlmoModel, OlmoForCausalLM } from './pre-trained-models/OlmoPreTrainedModel.js';
import { OpenELMModel, OpenELMForCausalLM } from './pre-trained-models/OpenELMPreTrainedModel.js';
import { OwlViTModel, OwlViTForObjectDetection } from './pre-trained-models/OwlViTPreTrainedModel.js';
import { Owlv2Model, Owlv2ForObjectDetection } from './pre-trained-models/Owlv2PreTrainedModel.js';
import { PaliGemmaForConditionalGeneration } from './pre-trained-models/PaliGemmaPreTrainedModel.js';
import { ParakeetForCTC } from './pre-trained-models/ParakeetPreTrainedModel.js';
import { PatchTSMixerModel, PatchTSMixerForPrediction } from './pre-trained-models/PatchTSMixerPreTrainedModel.js';
import { PatchTSTModel, PatchTSTForPrediction } from './pre-trained-models/PatchTSTPreTrainedModel.js';
import { Phi3Model, Phi3ForCausalLM } from './pre-trained-models/Phi3PreTrainedModel.js';
import { Phi3VForCausalLM } from './pre-trained-models/Phi3VPreTrainedModel.js';
import { PhiModel, PhiForCausalLM } from './pre-trained-models/PhiPreTrainedModel.js';
import { PvtModel, PvtForImageClassification } from './pre-trained-models/PvtPreTrainedModel.js';
import { PyAnnoteModel, PyAnnoteForAudioFrameClassification } from './pre-trained-models/PyAnnotePreTrainedModel.js';
import { Qwen2Model, Qwen2ForCausalLM } from './pre-trained-models/Qwen2PreTrainedModel.js';
import { Qwen2VLForConditionalGeneration } from './pre-trained-models/Qwen2VLPreTrainedModel.js';
import { Qwen3Model, Qwen3ForCausalLM } from './pre-trained-models/Qwen3PreTrainedModel.js';
import { RFDetrModel, RFDetrForObjectDetection } from './pre-trained-models/RFDetrPreTrainedModel.js';
import { RTDetrModel, RTDetrForObjectDetection } from './pre-trained-models/RTDetrPreTrainedModel.js';
import { RTDetrV2Model, RTDetrV2ForObjectDetection } from './pre-trained-models/RTDetrV2PreTrainedModel.js';
import { ResNetModel, ResNetForImageClassification } from './pre-trained-models/ResNetPreTrainedModel.js';
import {
    RoFormerModel,
    RoFormerForMaskedLM,
    RoFormerForSequenceClassification,
    RoFormerForTokenClassification,
    RoFormerForQuestionAnswering,
} from './pre-trained-models/RoFormerPreTrainedModel.js';
import {
    RobertaModel,
    RobertaForMaskedLM,
    RobertaForSequenceClassification,
    RobertaForTokenClassification,
    RobertaForQuestionAnswering,
} from './pre-trained-models/RobertaPreTrainedModel.js';
import { Sam2Model, EdgeTamModel, Sam3TrackerModel } from './pre-trained-models/Sam2PreTrainedModel.js';
import { SamModel } from './pre-trained-models/SamPreTrainedModel.js';
import {
    SapiensForSemanticSegmentation,
    SapiensForDepthEstimation,
    SapiensForNormalEstimation,
} from './pre-trained-models/SapiensPreTrainedModel.js';
import {
    SegformerForImageClassification,
    SegformerForSemanticSegmentation,
} from './pre-trained-models/SegformerPreTrainedModel.js';
import { SiglipModel, SiglipTextModel, SiglipVisionModel } from './pre-trained-models/SiglipPreTrainedModel.js';
import { SmolLM3Model, SmolLM3ForCausalLM } from './pre-trained-models/SmolLM3PreTrainedModel.js';
import { SnacModel, SnacEncoderModel, SnacDecoderModel } from './pre-trained-models/SnacPreTrainedModel.js';
import {
    SpeechT5ForSpeechToText,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
} from './pre-trained-models/SpeechT5PreTrainedModel.js';
import {
    SqueezeBertModel,
    SqueezeBertForMaskedLM,
    SqueezeBertForSequenceClassification,
    SqueezeBertForQuestionAnswering,
} from './pre-trained-models/SqueezeBertPreTrainedModel.js';
import { StableLmModel, StableLmForCausalLM } from './pre-trained-models/StableLmPreTrainedModel.js';
import { Starcoder2Model, Starcoder2ForCausalLM } from './pre-trained-models/Starcoder2PreTrainedModel.js';
import { StyleTextToSpeech2Model } from './pre-trained-models/StyleTextToSpeech2PreTrainedModel.js';
import { SupertonicForConditionalGeneration } from './pre-trained-models/SupertonicPreTrainedModel.js';
import { Swin2SRModel, Swin2SRForImageSuperResolution } from './pre-trained-models/Swin2SRPreTrainedModel.js';
import {
    SwinModel,
    SwinForImageClassification,
    SwinForSemanticSegmentation,
} from './pre-trained-models/SwinPreTrainedModel.js';
import { T5Model, T5ForConditionalGeneration } from './pre-trained-models/T5PreTrainedModel.js';
import {
    TableTransformerModel,
    TableTransformerForObjectDetection,
} from './pre-trained-models/TableTransformerPreTrainedModel.js';
import { TrOCRForCausalLM } from './pre-trained-models/TrOCRPreTrainedModel.js';
import { UltravoxModel, VoxtralForConditionalGeneration } from './pre-trained-models/UltravoxPreTrainedModel.js';
import {
    UniSpeechModel,
    UniSpeechForCTC,
    UniSpeechForSequenceClassification,
} from './pre-trained-models/UniSpeechPreTrainedModel.js';
import {
    UniSpeechSatModel,
    UniSpeechSatForCTC,
    UniSpeechSatForSequenceClassification,
    UniSpeechSatForAudioFrameClassification,
} from './pre-trained-models/UniSpeechSatPreTrainedModel.js';
import { VaultGemmaModel, VaultGemmaForCausalLM } from './pre-trained-models/VaultGemmaPreTrainedModel.js';
import { ViTMAEModel } from './pre-trained-models/ViTMAEPreTrainedModel.js';
import { ViTMSNModel, ViTMSNForImageClassification } from './pre-trained-models/ViTMSNPreTrainedModel.js';
import { ViTModel, ViTForImageClassification } from './pre-trained-models/ViTPreTrainedModel.js';
import { VisionEncoderDecoderModel } from './pre-trained-models/VisionEncoderDecoderModel.js';
import { VitMatteForImageMatting } from './pre-trained-models/VitMattePreTrainedModel.js';
import { VitPoseForPoseEstimation } from './pre-trained-models/VitPosePreTrainedModel.js';
import { VitsModel } from './pre-trained-models/VitsPreTrainedModel.js';
import {
    Wav2Vec2BertModel,
    Wav2Vec2BertForCTC,
    Wav2Vec2BertForSequenceClassification,
} from './pre-trained-models/Wav2Vec2BertPreTrainedModel.js';
import {
    Wav2Vec2Model,
    Wav2Vec2ForCTC,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2ForAudioFrameClassification,
} from './pre-trained-models/Wav2Vec2PreTrainedModel.js';
import {
    WavLMModel,
    WavLMForCTC,
    WavLMForSequenceClassification,
    WavLMForXVector,
    WavLMForAudioFrameClassification,
} from './pre-trained-models/WavLMPreTrainedModel.js';
import { WeSpeakerResNetModel } from './pre-trained-models/WeSpeakerResNetPreTrainedModel.js';
import {
    WhisperModel,
    WhisperForConditionalGeneration,
    LiteWhisperForConditionalGeneration,
} from './pre-trained-models/WhisperPreTrainedModel.js';
import {
    XLMModel,
    XLMWithLMHeadModel,
    XLMForSequenceClassification,
    XLMForTokenClassification,
    XLMForQuestionAnswering,
} from './pre-trained-models/XLMPreTrainedModel.js';
import {
    XLMRobertaModel,
    XLMRobertaForMaskedLM,
    XLMRobertaForSequenceClassification,
    XLMRobertaForTokenClassification,
    XLMRobertaForQuestionAnswering,
} from './pre-trained-models/XLMRobertaPreTrainedModel.js';
import { YolosModel, YolosForObjectDetection } from './pre-trained-models/YolosPreTrainedModel.js';


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

export {
    CUSTOM_ARCHITECTURES,
    MODEL_CLASS_TYPE_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES,
    MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES,
    MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_MASKED_LM_MAPPING_NAMES,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES,
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
};
