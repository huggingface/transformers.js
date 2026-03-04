/**
 * @file Pipeline task configurations and aliases
 *
 * Defines which pipeline class and model class(es) each pipeline task needs.
 * Tokenizer and processor loading is determined automatically from the model's files.
 */

import {
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForAudioClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSpeechSeq2Seq,
    AutoModelForTextToWaveform,
    AutoModelForTextToSpectrogram,
    AutoModelForCTC,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoModelForImageClassification,
    AutoModelForImageSegmentation,
    AutoModelForSemanticSegmentation,
    AutoModelForUniversalSegmentation,
    AutoModelForObjectDetection,
    AutoModelForZeroShotObjectDetection,
    AutoModelForDocumentQuestionAnswering,
    AutoModelForImageToImage,
    AutoModelForDepthEstimation,
    AutoModelForImageFeatureExtraction,
} from '../models/auto/modeling_auto.js';

import { TextClassificationPipeline } from './text-classification.js';
import { TokenClassificationPipeline } from './token-classification.js';
import { QuestionAnsweringPipeline } from './question-answering.js';
import { FillMaskPipeline } from './fill-mask.js';
import { SummarizationPipeline } from './summarization.js';
import { TranslationPipeline } from './translation.js';
import { Text2TextGenerationPipeline } from './text2text-generation.js';
import { TextGenerationPipeline } from './text-generation.js';
import { ZeroShotClassificationPipeline } from './zero-shot-classification.js';
import { AudioClassificationPipeline } from './audio-classification.js';
import { ZeroShotAudioClassificationPipeline } from './zero-shot-audio-classification.js';
import { AutomaticSpeechRecognitionPipeline } from './automatic-speech-recognition.js';
import { TextToAudioPipeline } from './text-to-audio.js';
import { ImageToTextPipeline } from './image-to-text.js';
import { ImageClassificationPipeline } from './image-classification.js';
import { ImageSegmentationPipeline } from './image-segmentation.js';
import { BackgroundRemovalPipeline } from './background-removal.js';
import { ZeroShotImageClassificationPipeline } from './zero-shot-image-classification.js';
import { ObjectDetectionPipeline } from './object-detection.js';
import { ZeroShotObjectDetectionPipeline } from './zero-shot-object-detection.js';
import { DocumentQuestionAnsweringPipeline } from './document-question-answering.js';
import { ImageToImagePipeline } from './image-to-image.js';
import { DepthEstimationPipeline } from './depth-estimation.js';
import { FeatureExtractionPipeline } from './feature-extraction.js';
import { ImageFeatureExtractionPipeline } from './image-feature-extraction.js';

export const SUPPORTED_TASKS = Object.freeze({
    'text-classification': {
        pipeline: TextClassificationPipeline,
        model: AutoModelForSequenceClassification,
        default: {
            model: 'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
        },
        type: 'text',
    },
    'token-classification': {
        pipeline: TokenClassificationPipeline,
        model: AutoModelForTokenClassification,
        default: {
            model: 'Xenova/bert-base-multilingual-cased-ner-hrl',
        },
        type: 'text',
    },
    'question-answering': {
        pipeline: QuestionAnsweringPipeline,
        model: AutoModelForQuestionAnswering,
        default: {
            model: 'Xenova/distilbert-base-cased-distilled-squad',
        },
        type: 'text',
    },
    'fill-mask': {
        pipeline: FillMaskPipeline,
        model: AutoModelForMaskedLM,
        default: {
            model: 'onnx-community/ettin-encoder-32m-ONNX',
            dtype: 'fp32',
        },
        type: 'text',
    },
    summarization: {
        pipeline: SummarizationPipeline,
        model: AutoModelForSeq2SeqLM,
        default: {
            model: 'Xenova/distilbart-cnn-6-6',
        },
        type: 'text',
    },
    translation: {
        pipeline: TranslationPipeline,
        model: AutoModelForSeq2SeqLM,
        default: {
            model: 'Xenova/t5-small',
        },
        type: 'text',
    },
    'text2text-generation': {
        pipeline: Text2TextGenerationPipeline,
        model: AutoModelForSeq2SeqLM,
        default: {
            model: 'Xenova/flan-t5-small',
        },
        type: 'text',
    },
    'text-generation': {
        pipeline: TextGenerationPipeline,
        model: AutoModelForCausalLM,
        default: {
            model: 'onnx-community/Qwen3-0.6B-ONNX',
            dtype: 'q4',
        },
        type: 'text',
    },
    'zero-shot-classification': {
        pipeline: ZeroShotClassificationPipeline,
        model: AutoModelForSequenceClassification,
        default: {
            model: 'Xenova/distilbert-base-uncased-mnli',
        },
        type: 'text',
    },
    'audio-classification': {
        pipeline: AudioClassificationPipeline,
        model: AutoModelForAudioClassification,
        default: {
            model: 'Xenova/wav2vec2-base-superb-ks',
        },
        type: 'audio',
    },
    'zero-shot-audio-classification': {
        pipeline: ZeroShotAudioClassificationPipeline,
        model: AutoModel,
        default: {
            model: 'Xenova/clap-htsat-unfused',
        },
        type: 'multimodal',
    },
    'automatic-speech-recognition': {
        pipeline: AutomaticSpeechRecognitionPipeline,
        model: [AutoModelForSpeechSeq2Seq, AutoModelForCTC],
        default: {
            model: 'Xenova/whisper-tiny.en',
        },
        type: 'multimodal',
    },
    'text-to-audio': {
        pipeline: TextToAudioPipeline,
        model: [AutoModelForTextToWaveform, AutoModelForTextToSpectrogram],
        default: {
            model: 'onnx-community/Supertonic-TTS-ONNX',
            dtype: 'fp32',
        },
        type: 'text',
    },
    'image-to-text': {
        pipeline: ImageToTextPipeline,
        model: AutoModelForVision2Seq,
        default: {
            model: 'Xenova/vit-gpt2-image-captioning',
        },
        type: 'multimodal',
    },
    'image-classification': {
        pipeline: ImageClassificationPipeline,
        model: AutoModelForImageClassification,
        default: {
            model: 'Xenova/vit-base-patch16-224',
        },
        type: 'multimodal',
    },
    'image-segmentation': {
        pipeline: ImageSegmentationPipeline,
        model: [AutoModelForImageSegmentation, AutoModelForSemanticSegmentation, AutoModelForUniversalSegmentation],
        default: {
            model: 'Xenova/detr-resnet-50-panoptic',
        },
        type: 'multimodal',
    },
    'background-removal': {
        pipeline: BackgroundRemovalPipeline,
        model: [AutoModelForImageSegmentation, AutoModelForSemanticSegmentation, AutoModelForUniversalSegmentation],
        default: {
            model: 'Xenova/modnet',
        },
        type: 'image',
    },
    'zero-shot-image-classification': {
        pipeline: ZeroShotImageClassificationPipeline,
        model: AutoModel,
        default: {
            model: 'Xenova/clip-vit-base-patch32',
        },
        type: 'multimodal',
    },
    'object-detection': {
        pipeline: ObjectDetectionPipeline,
        model: AutoModelForObjectDetection,
        default: {
            model: 'Xenova/detr-resnet-50',
        },
        type: 'multimodal',
    },
    'zero-shot-object-detection': {
        pipeline: ZeroShotObjectDetectionPipeline,
        model: AutoModelForZeroShotObjectDetection,
        default: {
            model: 'Xenova/owlvit-base-patch32',
        },
        type: 'multimodal',
    },
    'document-question-answering': {
        pipeline: DocumentQuestionAnsweringPipeline,
        model: AutoModelForDocumentQuestionAnswering,
        default: {
            model: 'Xenova/donut-base-finetuned-docvqa',
        },
        type: 'multimodal',
    },
    'image-to-image': {
        pipeline: ImageToImagePipeline,
        model: AutoModelForImageToImage,
        default: {
            model: 'Xenova/swin2SR-classical-sr-x2-64',
        },
        type: 'image',
    },
    'depth-estimation': {
        pipeline: DepthEstimationPipeline,
        model: AutoModelForDepthEstimation,
        default: {
            model: 'onnx-community/depth-anything-v2-small',
        },
        type: 'image',
    },
    'feature-extraction': {
        pipeline: FeatureExtractionPipeline,
        model: AutoModel,
        default: {
            model: 'onnx-community/all-MiniLM-L6-v2-ONNX',
            dtype: 'fp32',
        },
        type: 'text',
    },
    'image-feature-extraction': {
        pipeline: ImageFeatureExtractionPipeline,
        model: [AutoModelForImageFeatureExtraction, AutoModel],
        default: {
            model: 'onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX',
            dtype: 'fp32',
        },
        type: 'image',
    },
});

// TODO: Add types for TASK_ALIASES

export const TASK_ALIASES = Object.freeze({
    'sentiment-analysis': 'text-classification',
    ner: 'token-classification',
    // "vqa": "visual-question-answering", // TODO: Add
    asr: 'automatic-speech-recognition',
    'text-to-speech': 'text-to-audio',

    // Add for backwards compatibility
    embeddings: 'feature-extraction',
});

export {
    TextClassificationPipeline,
    TokenClassificationPipeline,
    QuestionAnsweringPipeline,
    FillMaskPipeline,
    SummarizationPipeline,
    TranslationPipeline,
    Text2TextGenerationPipeline,
    TextGenerationPipeline,
    ZeroShotClassificationPipeline,
    AudioClassificationPipeline,
    ZeroShotAudioClassificationPipeline,
    AutomaticSpeechRecognitionPipeline,
    TextToAudioPipeline,
    ImageToTextPipeline,
    ImageClassificationPipeline,
    ImageSegmentationPipeline,
    BackgroundRemovalPipeline,
    ZeroShotImageClassificationPipeline,
    ObjectDetectionPipeline,
    ZeroShotObjectDetectionPipeline,
    DocumentQuestionAnsweringPipeline,
    ImageToImagePipeline,
    DepthEstimationPipeline,
    FeatureExtractionPipeline,
    ImageFeatureExtractionPipeline,
};

/**
 * @typedef {keyof typeof SUPPORTED_TASKS} TaskType
 * @typedef {keyof typeof TASK_ALIASES} AliasType
 * @typedef {TaskType | AliasType} PipelineType All possible pipeline types.
 */
