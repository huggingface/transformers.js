/**
 * @file Pipelines provide a high-level, easy to use, API for running machine learning models.
 *
 * **Example:** Instantiate pipeline using the `pipeline` function.
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const classifier = await pipeline('sentiment-analysis');
 * const output = await classifier('I love transformers!');
 * // [{'label': 'POSITIVE', 'score': 0.999817686}]
 * ```
 *
 * @module pipelines
 */

import { AutoTokenizer } from './models/auto/tokenization_auto.js';
import { AutoProcessor } from './models/auto/processing_auto.js';
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
} from './models/auto/modeling_auto.js';

import { dispatchCallback } from './utils/core.js';

import { TextClassificationPipeline } from './pipelines/text-classification.js';
import { TokenClassificationPipeline } from './pipelines/token-classification.js';
import { QuestionAnsweringPipeline } from './pipelines/question-answering.js';
import { FillMaskPipeline } from './pipelines/fill-mask.js';
import { SummarizationPipeline } from './pipelines/summarization.js';
import { TranslationPipeline } from './pipelines/translation.js';
import { Text2TextGenerationPipeline } from './pipelines/text2text-generation.js';
import { TextGenerationPipeline } from './pipelines/text-generation.js';
import { ZeroShotClassificationPipeline } from './pipelines/zero-shot-classification.js';
import { AudioClassificationPipeline } from './pipelines/audio-classification.js';
import { ZeroShotAudioClassificationPipeline } from './pipelines/zero-shot-audio-classification.js';
import { AutomaticSpeechRecognitionPipeline } from './pipelines/automatic-speech-recognition.js';
import { TextToAudioPipeline } from './pipelines/text-to-audio.js';
import { ImageToTextPipeline } from './pipelines/image-to-text.js';
import { ImageClassificationPipeline } from './pipelines/image-classification.js';
import { ImageSegmentationPipeline } from './pipelines/image-segmentation.js';
import { BackgroundRemovalPipeline } from './pipelines/background-removal.js';
import { ZeroShotImageClassificationPipeline } from './pipelines/zero-shot-image-classification.js';
import { ObjectDetectionPipeline } from './pipelines/object-detection.js';
import { ZeroShotObjectDetectionPipeline } from './pipelines/zero-shot-object-detection.js';
import { DocumentQuestionAnsweringPipeline } from './pipelines/document-question-answering.js';
import { ImageToImagePipeline } from './pipelines/image-to-image.js';
import { DepthEstimationPipeline } from './pipelines/depth-estimation.js';
import { FeatureExtractionPipeline } from './pipelines/feature-extraction.js';
import { ImageFeatureExtractionPipeline } from './pipelines/image-feature-extraction.js';

const SUPPORTED_TASKS = Object.freeze({
    'text-classification': {
        tokenizer: AutoTokenizer,
        pipeline: TextClassificationPipeline,
        model: AutoModelForSequenceClassification,
        default: {
            // TODO: replace with original
            // "model": "distilbert-base-uncased-finetuned-sst-2-english",
            model: 'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
        },
        type: 'text',
    },
    'token-classification': {
        tokenizer: AutoTokenizer,
        pipeline: TokenClassificationPipeline,
        model: AutoModelForTokenClassification,
        default: {
            // TODO: replace with original
            // "model": "Davlan/bert-base-multilingual-cased-ner-hrl",
            model: 'Xenova/bert-base-multilingual-cased-ner-hrl',
        },
        type: 'text',
    },
    'question-answering': {
        tokenizer: AutoTokenizer,
        pipeline: QuestionAnsweringPipeline,
        model: AutoModelForQuestionAnswering,
        default: {
            // TODO: replace with original
            // "model": "distilbert-base-cased-distilled-squad",
            model: 'Xenova/distilbert-base-cased-distilled-squad',
        },
        type: 'text',
    },

    'fill-mask': {
        tokenizer: AutoTokenizer,
        pipeline: FillMaskPipeline,
        model: AutoModelForMaskedLM,
        default: {
            model: 'onnx-community/ettin-encoder-32m-ONNX',
            dtype: 'fp32',
        },
        type: 'text',
    },
    summarization: {
        tokenizer: AutoTokenizer,
        pipeline: SummarizationPipeline,
        model: AutoModelForSeq2SeqLM,
        default: {
            // TODO: replace with original
            // "model": "sshleifer/distilbart-cnn-6-6",
            model: 'Xenova/distilbart-cnn-6-6',
        },
        type: 'text',
    },
    translation: {
        tokenizer: AutoTokenizer,
        pipeline: TranslationPipeline,
        model: AutoModelForSeq2SeqLM,
        default: {
            // TODO: replace with original
            // "model": "t5-small",
            model: 'Xenova/t5-small',
        },
        type: 'text',
    },
    'text2text-generation': {
        tokenizer: AutoTokenizer,
        pipeline: Text2TextGenerationPipeline,
        model: AutoModelForSeq2SeqLM,
        default: {
            // TODO: replace with original
            // "model": "google/flan-t5-small",
            model: 'Xenova/flan-t5-small',
        },
        type: 'text',
    },
    'text-generation': {
        tokenizer: AutoTokenizer,
        pipeline: TextGenerationPipeline,
        model: AutoModelForCausalLM,
        default: {
            model: 'onnx-community/Qwen3-0.6B-ONNX',
            dtype: 'q4',
        },
        type: 'text',
    },
    'zero-shot-classification': {
        tokenizer: AutoTokenizer,
        pipeline: ZeroShotClassificationPipeline,
        model: AutoModelForSequenceClassification,
        default: {
            // TODO: replace with original
            // "model": "typeform/distilbert-base-uncased-mnli",
            model: 'Xenova/distilbert-base-uncased-mnli',
        },
        type: 'text',
    },
    'audio-classification': {
        pipeline: AudioClassificationPipeline,
        model: AutoModelForAudioClassification,
        processor: AutoProcessor,
        default: {
            // TODO: replace with original
            // "model": "superb/wav2vec2-base-superb-ks",
            model: 'Xenova/wav2vec2-base-superb-ks',
        },
        type: 'audio',
    },
    'zero-shot-audio-classification': {
        tokenizer: AutoTokenizer,
        pipeline: ZeroShotAudioClassificationPipeline,
        model: AutoModel,
        processor: AutoProcessor,
        default: {
            // TODO: replace with original
            // "model": "laion/clap-htsat-fused",
            model: 'Xenova/clap-htsat-unfused',
        },
        type: 'multimodal',
    },
    'automatic-speech-recognition': {
        tokenizer: AutoTokenizer,
        pipeline: AutomaticSpeechRecognitionPipeline,
        model: [AutoModelForSpeechSeq2Seq, AutoModelForCTC],
        processor: AutoProcessor,
        default: {
            // TODO: replace with original
            // "model": "openai/whisper-tiny.en",
            model: 'Xenova/whisper-tiny.en',
        },
        type: 'multimodal',
    },
    'text-to-audio': {
        tokenizer: AutoTokenizer,
        pipeline: TextToAudioPipeline,
        model: [AutoModelForTextToWaveform, AutoModelForTextToSpectrogram],
        processor: [AutoProcessor, /* Some don't use a processor */ null],
        default: {
            model: 'onnx-community/Supertonic-TTS-ONNX',
            dtype: 'fp32',
        },
        type: 'text',
    },
    'image-to-text': {
        tokenizer: AutoTokenizer,
        pipeline: ImageToTextPipeline,
        model: AutoModelForVision2Seq,
        processor: AutoProcessor,
        default: {
            // TODO: replace with original
            // "model": "nlpconnect/vit-gpt2-image-captioning",
            model: 'Xenova/vit-gpt2-image-captioning',
        },
        type: 'multimodal',
    },

    'image-classification': {
        // no tokenizer
        pipeline: ImageClassificationPipeline,
        model: AutoModelForImageClassification,
        processor: AutoProcessor,
        default: {
            // TODO: replace with original
            // "model": "google/vit-base-patch16-224",
            model: 'Xenova/vit-base-patch16-224',
        },
        type: 'multimodal',
    },

    'image-segmentation': {
        // no tokenizer
        pipeline: ImageSegmentationPipeline,
        model: [AutoModelForImageSegmentation, AutoModelForSemanticSegmentation, AutoModelForUniversalSegmentation],
        processor: AutoProcessor,
        default: {
            // TODO: replace with original
            // "model": "facebook/detr-resnet-50-panoptic",
            model: 'Xenova/detr-resnet-50-panoptic',
        },
        type: 'multimodal',
    },
    'background-removal': {
        // no tokenizer
        pipeline: BackgroundRemovalPipeline,
        model: [AutoModelForImageSegmentation, AutoModelForSemanticSegmentation, AutoModelForUniversalSegmentation],
        processor: AutoProcessor,
        default: {
            model: 'Xenova/modnet',
        },
        type: 'image',
    },

    'zero-shot-image-classification': {
        tokenizer: AutoTokenizer,
        pipeline: ZeroShotImageClassificationPipeline,
        model: AutoModel,
        processor: AutoProcessor,
        default: {
            // TODO: replace with original
            // "model": "openai/clip-vit-base-patch32",
            model: 'Xenova/clip-vit-base-patch32',
        },
        type: 'multimodal',
    },

    'object-detection': {
        // no tokenizer
        pipeline: ObjectDetectionPipeline,
        model: AutoModelForObjectDetection,
        processor: AutoProcessor,
        default: {
            // TODO: replace with original
            // "model": "facebook/detr-resnet-50",
            model: 'Xenova/detr-resnet-50',
        },
        type: 'multimodal',
    },
    'zero-shot-object-detection': {
        tokenizer: AutoTokenizer,
        pipeline: ZeroShotObjectDetectionPipeline,
        model: AutoModelForZeroShotObjectDetection,
        processor: AutoProcessor,
        default: {
            // TODO: replace with original
            // "model": "google/owlvit-base-patch32",
            model: 'Xenova/owlvit-base-patch32',
        },
        type: 'multimodal',
    },
    'document-question-answering': {
        tokenizer: AutoTokenizer,
        pipeline: DocumentQuestionAnsweringPipeline,
        model: AutoModelForDocumentQuestionAnswering,
        processor: AutoProcessor,
        default: {
            // TODO: replace with original
            // "model": "naver-clova-ix/donut-base-finetuned-docvqa",
            model: 'Xenova/donut-base-finetuned-docvqa',
        },
        type: 'multimodal',
    },
    'image-to-image': {
        // no tokenizer
        pipeline: ImageToImagePipeline,
        model: AutoModelForImageToImage,
        processor: AutoProcessor,
        default: {
            // TODO: replace with original
            // "model": "caidas/swin2SR-classical-sr-x2-64",
            model: 'Xenova/swin2SR-classical-sr-x2-64',
        },
        type: 'image',
    },
    'depth-estimation': {
        // no tokenizer
        pipeline: DepthEstimationPipeline,
        model: AutoModelForDepthEstimation,
        processor: AutoProcessor,
        default: {
            model: 'onnx-community/depth-anything-v2-small',
        },
        type: 'image',
    },

    // This task serves as a useful interface for dealing with sentence-transformers (https://huggingface.co/sentence-transformers).
    'feature-extraction': {
        tokenizer: AutoTokenizer,
        pipeline: FeatureExtractionPipeline,
        model: AutoModel,
        default: {
            model: 'onnx-community/all-MiniLM-L6-v2-ONNX',
            dtype: 'fp32',
        },
        type: 'text',
    },
    'image-feature-extraction': {
        processor: AutoProcessor,
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
const TASK_ALIASES = Object.freeze({
    'sentiment-analysis': 'text-classification',
    ner: 'token-classification',
    // "vqa": "visual-question-answering", // TODO: Add
    asr: 'automatic-speech-recognition',
    'text-to-speech': 'text-to-audio',

    // Add for backwards compatibility
    embeddings: 'feature-extraction',
});

/**
 * @typedef {keyof typeof SUPPORTED_TASKS} TaskType
 * @typedef {keyof typeof TASK_ALIASES} AliasType
 * @typedef {TaskType | AliasType} PipelineType All possible pipeline types.
 * @typedef {{[K in TaskType]: InstanceType<typeof SUPPORTED_TASKS[K]["pipeline"]>}} SupportedTasks A mapping of pipeline names to their corresponding pipeline classes.
 * @typedef {{[K in AliasType]: InstanceType<typeof SUPPORTED_TASKS[TASK_ALIASES[K]]["pipeline"]>}} AliasTasks A mapping from pipeline aliases to their corresponding pipeline classes.
 * @typedef {SupportedTasks & AliasTasks} AllTasks A mapping from all pipeline names and aliases to their corresponding pipeline classes.
 */

/**
 * Utility factory method to build a `Pipeline` object.
 *
 * @template {PipelineType} T The type of pipeline to return.
 * @param {T} task The task defining which pipeline will be returned. Currently accepted tasks are:
 *  - `"audio-classification"`: will return a `AudioClassificationPipeline`.
 *  - `"automatic-speech-recognition"`: will return a `AutomaticSpeechRecognitionPipeline`.
 *  - `"depth-estimation"`: will return a `DepthEstimationPipeline`.
 *  - `"document-question-answering"`: will return a `DocumentQuestionAnsweringPipeline`.
 *  - `"feature-extraction"`: will return a `FeatureExtractionPipeline`.
 *  - `"fill-mask"`: will return a `FillMaskPipeline`.
 *  - `"image-classification"`: will return a `ImageClassificationPipeline`.
 *  - `"image-segmentation"`: will return a `ImageSegmentationPipeline`.
 *  - `"image-to-text"`: will return a `ImageToTextPipeline`.
 *  - `"object-detection"`: will return a `ObjectDetectionPipeline`.
 *  - `"question-answering"`: will return a `QuestionAnsweringPipeline`.
 *  - `"summarization"`: will return a `SummarizationPipeline`.
 *  - `"text2text-generation"`: will return a `Text2TextGenerationPipeline`.
 *  - `"text-classification"` (alias "sentiment-analysis" available): will return a `TextClassificationPipeline`.
 *  - `"text-generation"`: will return a `TextGenerationPipeline`.
 *  - `"token-classification"` (alias "ner" available): will return a `TokenClassificationPipeline`.
 *  - `"translation"`: will return a `TranslationPipeline`.
 *  - `"translation_xx_to_yy"`: will return a `TranslationPipeline`.
 *  - `"zero-shot-classification"`: will return a `ZeroShotClassificationPipeline`.
 *  - `"zero-shot-audio-classification"`: will return a `ZeroShotAudioClassificationPipeline`.
 *  - `"zero-shot-image-classification"`: will return a `ZeroShotImageClassificationPipeline`.
 *  - `"zero-shot-object-detection"`: will return a `ZeroShotObjectDetectionPipeline`.
 * @param {string} [model=null] The name of the pre-trained model to use. If not specified, the default model for the task will be used.
 * @param {import('./utils/hub.js').PretrainedModelOptions} [options] Optional parameters for the pipeline.
 * @returns {Promise<AllTasks[T]>} A Pipeline object for the specified task.
 * @throws {Error} If an unsupported pipeline is requested.
 */
export async function pipeline(
    task,
    model = null,
    {
        progress_callback = null,
        config = null,
        cache_dir = null,
        local_files_only = false,
        revision = 'main',
        device = null,
        dtype = null,
        subfolder = 'onnx',
        use_external_data_format = null,
        model_file_name = null,
        session_options = {},
    } = {},
) {
    // Helper method to construct pipeline

    // Apply aliases
    // @ts-ignore
    task = TASK_ALIASES[task] ?? task;

    // Get pipeline info
    const pipelineInfo = SUPPORTED_TASKS[task.split('_', 1)[0]];
    if (!pipelineInfo) {
        throw Error(`Unsupported pipeline: ${task}. Must be one of [${Object.keys(SUPPORTED_TASKS)}]`);
    }

    // Use model if specified, otherwise, use default
    if (!model) {
        model = pipelineInfo.default.model;
        console.log(`No model specified. Using default model: "${model}".`);
        if (!dtype && pipelineInfo.default.dtype) {
            dtype = pipelineInfo.default.dtype;
        }
    }

    const pretrainedOptions = {
        progress_callback,
        config,
        cache_dir,
        local_files_only,
        revision,
        device,
        dtype,
        subfolder,
        use_external_data_format,
        model_file_name,
        session_options,
    };

    const classes = new Map([
        ['tokenizer', pipelineInfo.tokenizer],
        ['model', pipelineInfo.model],
        ['processor', pipelineInfo.processor],
    ]);

    // Load model, tokenizer, and processor (if they exist)
    const results = await loadItems(classes, model, pretrainedOptions);
    results.task = task;

    dispatchCallback(progress_callback, {
        status: 'ready',
        task: task,
        model: model,
    });

    const pipelineClass = pipelineInfo.pipeline;
    return new pipelineClass(results);
}

/**
 * Helper function to get applicable model, tokenizer, or processor classes for a given model.
 * @param {Map<string, any>} mapping The mapping of names to classes, arrays of classes, or null.
 * @param {string} model The name of the model to load.
 * @param {import('./utils/hub.js').PretrainedOptions} pretrainedOptions The options to pass to the `from_pretrained` method.
 * @private
 */
async function loadItems(mapping, model, pretrainedOptions) {
    const result = Object.create(null);

    /**@type {Promise[]} */
    const promises = [];
    for (const [name, cls] of mapping.entries()) {
        if (!cls) continue;

        /**@type {Promise} */
        let promise;
        if (Array.isArray(cls)) {
            promise = new Promise(async (resolve, reject) => {
                let e;
                for (const c of cls) {
                    if (c === null) {
                        // If null, we resolve it immediately, meaning the relevant
                        // class was not found, but it is optional.
                        resolve(null);
                        return;
                    }
                    try {
                        resolve(await c.from_pretrained(model, pretrainedOptions));
                        return;
                    } catch (err) {
                        if (err.message?.includes('Unsupported model type')) {
                            // If the error is due to an unsupported model type, we
                            // save the error and try the next class.
                            e = err;
                        } else if (err.message?.includes('Could not locate file')) {
                            e = err;
                        } else {
                            reject(err);
                            return;
                        }
                    }
                }
                reject(e);
            });
        } else {
            promise = cls.from_pretrained(model, pretrainedOptions);
        }

        result[name] = promise;
        promises.push(promise);
    }

    // Wait for all promises to resolve (in parallel)
    await Promise.all(promises);

    // Then assign to result
    for (const [name, promise] of Object.entries(result)) {
        result[name] = await promise;
    }

    return result;
}

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
