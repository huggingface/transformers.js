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

import { DefaultProgressCallback, dispatchCallback } from './utils/core.js';
import { logger } from './utils/logger.js';

import { AutoTokenizer } from './models/auto/tokenization_auto.js';
import { AutoProcessor } from './models/auto/processing_auto.js';
import { AutoConfig } from './configs.js';

import {
    SUPPORTED_TASKS,
    TASK_ALIASES,
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
} from './pipelines/index.js';
import { get_pipeline_files } from './utils/model_registry/get_pipeline_files.js';
import { get_file_metadata } from './utils/model_registry/get_file_metadata.js';

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
 *  - `"background-removal"`: will return a `BackgroundRemovalPipeline`.
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
        logger.info(`No model specified. Using default model: "${model}".`);
        if (!dtype && pipelineInfo.default.dtype) {
            dtype = pipelineInfo.default.dtype;
        }
    }

    // Determine which files the model needs
    const expected_files = await get_pipeline_files(task, model, {
        device,
        dtype,
    });

    /** @type {import('./utils/core.js').FilesLoadingMap} */
    let files_loading = {};
    if (progress_callback) {
        /** @type {Array<{exists: boolean, size?: number, contentType?: string, fromCache?: boolean}>} */
        const metadata = await Promise.all(expected_files.map(async (file) => get_file_metadata(model, file)));
        metadata.forEach((m, i) => {
            if (m.exists) {
                files_loading[expected_files[i]] = {
                    loaded: 0,
                    total: m.size ?? 0,
                };
            }
        });
    }

    const pretrainedOptions = {
        progress_callback: progress_callback
            ? new DefaultProgressCallback(progress_callback, files_loading)
            : undefined,
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

    // Determine which components to load based on the expected files
    const hasTokenizer = expected_files.includes('tokenizer.json');
    const hasProcessor = expected_files.includes('preprocessor_config.json');

    // Resolve the correct model class (needs config when multiple candidates exist)
    const modelClasses = pipelineInfo.model;
    let modelPromise;
    if (Array.isArray(modelClasses)) {
        const resolvedConfig = config ?? (await AutoConfig.from_pretrained(model, pretrainedOptions));
        const { model_type } = resolvedConfig;
        const matchedClass = modelClasses.find((cls) => cls.supports(model_type));
        if (!matchedClass) {
            throw Error(
                `Unsupported model type "${model_type}" for task "${task}". ` +
                    `None of the candidate model classes support this type.`,
            );
        }
        modelPromise = matchedClass.from_pretrained(model, { ...pretrainedOptions, config: resolvedConfig });
    } else {
        modelPromise = modelClasses.from_pretrained(model, pretrainedOptions);
    }

    // Load all components in parallel
    const [tokenizer, processor, model_loaded] = await Promise.all([
        hasTokenizer ? AutoTokenizer.from_pretrained(model, pretrainedOptions) : null,
        hasProcessor ? AutoProcessor.from_pretrained(model, pretrainedOptions) : null,
        modelPromise,
    ]);

    const results = { task, model: model_loaded };
    if (tokenizer) results.tokenizer = tokenizer;
    if (processor) results.processor = processor;

    dispatchCallback(progress_callback, {
        status: 'ready',
        task: task,
        model: model,
    });

    const pipelineClass = pipelineInfo.pipeline;
    return new pipelineClass(results);
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

// Export pipeline output types
/**
 * @typedef {import('./pipelines/fill-mask.js').FillMaskOutput} FillMaskOutput
 * @typedef {import('./pipelines/text-classification.js').TextClassificationOutput} TextClassificationOutput
 * @typedef {import('./pipelines/token-classification.js').TokenClassificationOutput} TokenClassificationOutput
 * @typedef {import('./pipelines/question-answering.js').QuestionAnsweringOutput} QuestionAnsweringOutput
 * @typedef {import('./pipelines/summarization.js').SummarizationOutput} SummarizationOutput
 * @typedef {import('./pipelines/translation.js').TranslationOutput} TranslationOutput
 * @typedef {import('./pipelines/text2text-generation.js').Text2TextGenerationOutput} Text2TextGenerationOutput
 * @typedef {import('./pipelines/text-generation.js').TextGenerationOutput} TextGenerationOutput
 * @typedef {import('./pipelines/text-generation.js').TextGenerationStringOutput} TextGenerationStringOutput
 * @typedef {import('./pipelines/text-generation.js').TextGenerationChatOutput} TextGenerationChatOutput
 * @typedef {import('./pipelines/zero-shot-classification.js').ZeroShotClassificationOutput} ZeroShotClassificationOutput
 * @typedef {import('./pipelines/audio-classification.js').AudioClassificationOutput} AudioClassificationOutput
 * @typedef {import('./pipelines/zero-shot-audio-classification.js').ZeroShotAudioClassificationOutput} ZeroShotAudioClassificationOutput
 * @typedef {import('./pipelines/automatic-speech-recognition.js').AutomaticSpeechRecognitionOutput} AutomaticSpeechRecognitionOutput
 * @typedef {import('./pipelines/text-to-audio.js').TextToAudioOutput} TextToAudioOutput
 * @typedef {import('./pipelines/image-classification.js').ImageClassificationOutput} ImageClassificationOutput
 * @typedef {import('./pipelines/image-segmentation.js').ImageSegmentationOutput} ImageSegmentationOutput
 * @typedef {import('./pipelines/image-to-text.js').ImageToTextOutput} ImageToTextOutput
 * @typedef {import('./pipelines/object-detection.js').ObjectDetectionOutput} ObjectDetectionOutput
 * @typedef {import('./pipelines/zero-shot-object-detection.js').ZeroShotObjectDetectionOutput} ZeroShotObjectDetectionOutput
 * @typedef {import('./pipelines/zero-shot-image-classification.js').ZeroShotImageClassificationOutput} ZeroShotImageClassificationOutput
 * @typedef {import('./pipelines/document-question-answering.js').DocumentQuestionAnsweringOutput} DocumentQuestionAnsweringOutput
 * @typedef {import('./pipelines/depth-estimation.js').DepthEstimationOutput} DepthEstimationOutput
 */
