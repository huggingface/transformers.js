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
import { SUPPORTED_TASKS, TASK_ALIASES } from './pipelines/index.js';
import { get_pipeline_files } from './utils/cache/get_pipeline_files.js';
import { get_file_metadata } from './utils/cache/get_file_metadata.js';

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

    const expected_files = Boolean(progress_callback)
        ? await get_pipeline_files(task, model, {
              device,
              dtype,
          })
        : [];

    /** @type {import('./utils/core.js').FilesLoadingMap} */
    let files_loading = {};
    if (Boolean(progress_callback)) {
        /** @type {Array<{exists: boolean, size?: number, contentType?: string, fromCache?: boolean}>} */
        const metadata = await Promise.all(expected_files.map(async (file) => get_file_metadata(model, file)));
        metadata.map((m, i) => {
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
            ? /** @param {import('./utils/core.js').ProgressInfo} info */
              (info) => {
                  if (info.status === 'progress') {
                      files_loading[info.file] = {
                          loaded: info.loaded,
                          total: info.total,
                      };

                      const loaded = Object.values(files_loading).reduce((acc, curr) => acc + curr.loaded, 0);
                      const total = Object.values(files_loading).reduce((acc, curr) => acc + curr.total, 0);
                      const progress = (loaded / total) * 100;

                      progress_callback({
                          status: 'progress_total',
                          name: info.name,
                          progress,
                          loaded,
                          total,
                          files: structuredClone(files_loading),
                      });
                  }
                  progress_callback(info);
              }
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
