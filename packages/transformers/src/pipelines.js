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
import {
    getModelId,
    isInferenceBackend,
    isOnnxSessionProvider,
    loadInferenceModel,
    withInferenceBackendSharedAssetOptions,
    validateInferenceBackendTask,
    validateInferenceModelTask,
} from './backends/inference.js';
import { validateInferenceArtifactProvider } from './backends/artifacts.js';
import { getModelJSON, getModelText } from './utils/hub.js';
import { CHAT_TEMPLATE_NAME } from './utils/constants.js';

/**
 * @typedef {keyof typeof SUPPORTED_TASKS} TaskType
 * @typedef {keyof typeof TASK_ALIASES} AliasType
 * @typedef {TaskType | AliasType} PipelineType All possible pipeline types.
 * @typedef {{[K in TaskType]: InstanceType<typeof SUPPORTED_TASKS[K]["pipeline"]>}} SupportedTasks A mapping of pipeline names to their corresponding pipeline classes.
 * @typedef {{[K in AliasType]: InstanceType<typeof SUPPORTED_TASKS[TASK_ALIASES[K]]["pipeline"]>}} AliasTasks A mapping from pipeline aliases to their corresponding pipeline classes.
 * @typedef {SupportedTasks & AliasTasks} AllTasks A mapping from all pipeline names and aliases to their corresponding pipeline classes.
 */

/**
 * Resolve a custom backend's default chat template without moving rendering policy into the backend.
 *
 * @param {import('./backends/inference.js').InferenceBackend} backend
 * @param {import('./utils/hub.js').PretrainedModelOptions} options
 * @param {import('./utils/hub.js').PretrainedModelOptions} [loadLocation]
 * @returns {Promise<string>|null}
 */
export function loadInferenceBackendChatTemplate(backend, options, loadLocation = options) {
    const source = backend.chatTemplate;
    if (source == null) return null;
    if (typeof source !== 'object') {
        throw new TypeError('Inference backend `chatTemplate` must be an object.');
    }
    if (Object.hasOwn(source, 'content')) {
        if (
            typeof source.content !== 'string' ||
            ['modelId', 'revision', 'subfolder', 'file'].some((key) => Object.hasOwn(source, key))
        ) {
            throw new TypeError(
                'Inference backend `chatTemplate.content` must be a string and cannot be combined with source fields.',
            );
        }
        return Promise.resolve(source.content);
    }
    if (source.modelId !== undefined && typeof source.modelId !== 'string') {
        throw new TypeError('Inference backend `chatTemplate.modelId` must be a string.');
    }
    if (source.file !== undefined && typeof source.file !== 'string') {
        throw new TypeError('Inference backend `chatTemplate.file` must be a string.');
    }
    if (source.revision !== undefined && typeof source.revision !== 'string') {
        throw new TypeError('Inference backend `chatTemplate.revision` must be a string.');
    }
    if (source.subfolder !== undefined && typeof source.subfolder !== 'string') {
        throw new TypeError('Inference backend `chatTemplate.subfolder` must be a string.');
    }
    const usesSharedRepository = source.modelId === undefined || source.modelId === backend.modelId;
    const location = usesSharedRepository ? options : loadLocation;
    return getModelText(source.modelId ?? backend.modelId, source.file ?? CHAT_TEMPLATE_NAME, true, {
        ...location,
        revision: source.revision ?? location.revision,
        subfolder: source.subfolder ?? location.subfolder,
    });
}

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
 * @param {string|import('./backends/inference.js').InferenceBackend} [model=null] The model ID or custom inference backend to use. If not specified, the default model for the task will be used.
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
        subfolder = null,
        use_external_data_format = null,
        model_file_name = null,
        session_options = {},
        signal = undefined,
        artifactProvider = undefined,
    } = {},
) {
    const loadLocation = { revision, subfolder };
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

    const customBackend = isInferenceBackend(model) && !isOnnxSessionProvider(model);
    const customRegistryProvider =
        customBackend && typeof (/** @type {any} */ (model).listModelArtifacts) === 'function'
            ? /** @type {import('./backends/model_registry.js').ModelRegistryInferenceProvider} */ (model)
            : null;
    const modelId = getModelId(model);
    if (customBackend) {
        ({ revision, subfolder } = withInferenceBackendSharedAssetOptions(
            /** @type {import('./backends/inference.js').InferenceBackend} */ (model),
            { revision, subfolder },
        ));
    }
    validateInferenceArtifactProvider(artifactProvider);
    if (customBackend) {
        validateInferenceBackendTask(/** @type {import('./backends/inference.js').InferenceBackend} */ (model), task);
    }

    // Determine which files the model needs
    const expected_files = await get_pipeline_files(task, modelId, {
        device,
        dtype,
        config,
        cache_dir,
        local_files_only,
        revision,
        subfolder,
        signal,
        use_external_data_format,
        model_file_name,
        inferenceProvider: customRegistryProvider,
        include_model: !customBackend || customRegistryProvider !== null,
    });

    /** @type {import('./utils/core.js').FilesLoadingMap} */
    let files_loading = {};
    /** @type {Record<string, {size?: number, fromCache?: boolean}>} */
    const artifactMetadata = {};
    if (progress_callback) {
        /** @type {Array<{exists: boolean, size?: number, contentType?: string, fromCache?: boolean}>} */
        const metadata = await Promise.all(
            expected_files.map(async (file) =>
                get_file_metadata(customBackend ? model : modelId, file, {
                    cache_dir,
                    local_files_only,
                    revision,
                    subfolder,
                    signal,
                }),
            ),
        );
        metadata.forEach((m, i) => {
            if (m.exists) {
                artifactMetadata[expected_files[i]] = { size: m.size, fromCache: m.fromCache };
                files_loading[expected_files[i]] = {
                    loaded: m.fromCache ? (m.size ?? 0) : 0,
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
        generation_config: null,
        signal,
        artifactProvider,
        artifactMetadata,
    };

    // Determine which components to load based on the expected files
    const hasTokenizer = expected_files.includes('tokenizer.json');
    const hasProcessor = expected_files.includes('preprocessor_config.json');

    // Resolve the correct model class (needs config when multiple candidates exist)
    const modelClasses = pipelineInfo.model;
    let modelPromise;
    if (customBackend) {
        pretrainedOptions.config = config ?? (await AutoConfig.from_pretrained(modelId, pretrainedOptions));
        if (expected_files.includes('generation_config.json')) {
            pretrainedOptions.generation_config = await getModelJSON(
                modelId,
                'generation_config.json',
                false,
                pretrainedOptions,
            );
        }
        modelPromise = loadInferenceModel(/** @type {import('./backends/inference.js').InferenceBackend} */ (model), {
            ...pretrainedOptions,
            task,
        });
    } else if (Array.isArray(modelClasses)) {
        const resolvedConfig = config ?? (await AutoConfig.from_pretrained(modelId, pretrainedOptions));
        const { model_type } = resolvedConfig;
        const matchedClass = modelClasses.find((cls) => cls.supports(model_type));
        if (!matchedClass) {
            throw Error(
                `Unsupported model type "${model_type}" for task "${task}". ` +
                    `None of the candidate model classes support this type.`,
            );
        }
        modelPromise = matchedClass.from_pretrained(modelId, { ...pretrainedOptions, config: resolvedConfig });
    } else {
        modelPromise = modelClasses.from_pretrained(modelId, pretrainedOptions);
    }

    let tokenizer;
    let processor;
    let model_loaded;
    let chat_template;
    try {
        // Load all components in parallel.
        [tokenizer, processor, model_loaded, chat_template] = await Promise.all([
            hasTokenizer ? AutoTokenizer.from_pretrained(modelId, pretrainedOptions) : null,
            hasProcessor ? AutoProcessor.from_pretrained(modelId, pretrainedOptions) : null,
            modelPromise,
            customBackend && hasTokenizer
                ? loadInferenceBackendChatTemplate(
                      /** @type {import('./backends/inference.js').InferenceBackend} */ (model),
                      pretrainedOptions,
                      loadLocation,
                  )
                : null,
        ]);
        if (tokenizer && chat_template != null) tokenizer.chat_template = chat_template;
        if (customBackend) {
            validateInferenceModelTask(model_loaded, task);
        }
    } catch (error) {
        // A parallel tokenizer/processor failure may race with a successful GPU model load.
        const loadedModel = model_loaded ?? (await modelPromise.catch(() => null));
        try {
            await loadedModel?.dispose?.();
        } catch {}
        throw error;
    }

    const results = { task, model: model_loaded };
    if (tokenizer) results.tokenizer = tokenizer;
    if (processor) results.processor = processor;

    dispatchCallback(progress_callback, {
        status: 'ready',
        task: task,
        model: modelId,
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
