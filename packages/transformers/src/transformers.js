/**
 * @file Entry point for the Transformers.js library. Only the exports from this file
 * are available to the end user, and are grouped as follows:
 *
 * 1. [Environment variables](./env)
 * 2. [Pipelines](./pipelines)
 * 3. [Models](./models)
 * 4. [Tokenizers](./tokenizers)
 * 5. [Processors](./processors)
 * 6. [Configs](./configs)
 *
 * @module transformers
 */

// Environment variables
export { env } from './env.js';

// Pipelines
export * from './pipelines.js';

/**
 * @typedef {import('./pipelines/fill-mask.js').FillMaskSingle} FillMaskSingle
 * @typedef {import('./pipelines/fill-mask.js').FillMaskOutput} FillMaskOutput
 * @typedef {import('./pipelines/text-classification.js').TextClassificationSingle} TextClassificationSingle
 * @typedef {import('./pipelines/text-classification.js').TextClassificationOutput} TextClassificationOutput
 * @typedef {import('./pipelines/token-classification.js').TokenClassificationSingle} TokenClassificationSingle
 * @typedef {import('./pipelines/token-classification.js').TokenClassificationOutput} TokenClassificationOutput
 * @typedef {import('./pipelines/question-answering.js').QuestionAnsweringOutput} QuestionAnsweringOutput
 * @typedef {import('./pipelines/summarization.js').SummarizationSingle} SummarizationSingle
 * @typedef {import('./pipelines/summarization.js').SummarizationOutput} SummarizationOutput
 * @typedef {import('./pipelines/translation.js').TranslationSingle} TranslationSingle
 * @typedef {import('./pipelines/translation.js').TranslationOutput} TranslationOutput
 * @typedef {import('./pipelines/text2text-generation.js').Text2TextGenerationSingle} Text2TextGenerationSingle
 * @typedef {import('./pipelines/text2text-generation.js').Text2TextGenerationOutput} Text2TextGenerationOutput
 * @typedef {import('./pipelines/text-generation.js').TextGenerationSingle} TextGenerationSingle
 * @typedef {import('./pipelines/text-generation.js').TextGenerationOutput} TextGenerationOutput
 * @typedef {import('./pipelines/text-generation.js').TextGenerationStringOutput} TextGenerationStringOutput
 * @typedef {import('./pipelines/text-generation.js').TextGenerationChatOutput} TextGenerationChatOutput
 * @typedef {import('./pipelines/zero-shot-classification.js').ZeroShotClassificationOutput} ZeroShotClassificationOutput
 * @typedef {import('./pipelines/audio-classification.js').AudioClassificationSingle} AudioClassificationSingle
 * @typedef {import('./pipelines/audio-classification.js').AudioClassificationOutput} AudioClassificationOutput
 * @typedef {import('./pipelines/zero-shot-audio-classification.js').ZeroShotAudioClassificationOutputSingle} ZeroShotAudioClassificationOutputSingle
 * @typedef {import('./pipelines/zero-shot-audio-classification.js').ZeroShotAudioClassificationOutput} ZeroShotAudioClassificationOutput
 * @typedef {import('./pipelines/automatic-speech-recognition.js').AutomaticSpeechRecognitionOutput} AutomaticSpeechRecognitionOutput
 * @typedef {import('./pipelines/text-to-audio.js').TextToAudioOutput} TextToAudioOutput
 * @typedef {import('./pipelines/image-classification.js').ImageClassificationSingle} ImageClassificationSingle
 * @typedef {import('./pipelines/image-classification.js').ImageClassificationOutput} ImageClassificationOutput
 * @typedef {import('./pipelines/image-segmentation.js').ImageSegmentationOutputSingle} ImageSegmentationOutputSingle
 * @typedef {import('./pipelines/image-segmentation.js').ImageSegmentationOutput} ImageSegmentationOutput
 * @typedef {import('./pipelines/image-to-text.js').ImageToTextSingle} ImageToTextSingle
 * @typedef {import('./pipelines/image-to-text.js').ImageToTextOutput} ImageToTextOutput
 * @typedef {import('./pipelines/object-detection.js').ObjectDetectionPipelineSingle} ObjectDetectionPipelineSingle
 * @typedef {import('./pipelines/object-detection.js').ObjectDetectionOutput} ObjectDetectionOutput
 * @typedef {import('./pipelines/zero-shot-object-detection.js').ZeroShotObjectDetectionOutputSingle} ZeroShotObjectDetectionOutputSingle
 * @typedef {import('./pipelines/zero-shot-object-detection.js').ZeroShotObjectDetectionOutput} ZeroShotObjectDetectionOutput
 * @typedef {import('./pipelines/zero-shot-image-classification.js').ZeroShotImageClassificationOutputSingle} ZeroShotImageClassificationOutputSingle
 * @typedef {import('./pipelines/zero-shot-image-classification.js').ZeroShotImageClassificationOutput} ZeroShotImageClassificationOutput
 * @typedef {import('./pipelines/document-question-answering.js').DocumentQuestionAnsweringSingle} DocumentQuestionAnsweringSingle
 * @typedef {import('./pipelines/document-question-answering.js').DocumentQuestionAnsweringOutput} DocumentQuestionAnsweringOutput
 * @typedef {import('./pipelines/depth-estimation.js').DepthEstimationOutput} DepthEstimationOutput
 */

// Models
export * from './models/models.js';
export * from './models/auto/modeling_auto.js';

// Tokenizers
export * from './models/tokenizers.js';
export * from './models/auto/tokenization_auto.js';

// Feature Extractors
export * from './models/feature_extractors.js';
export * from './models/auto/feature_extraction_auto.js';

// Image Processors
export * from './models/image_processors.js';
export * from './models/auto/image_processing_auto.js';

// Processors
export * from './models/processors.js';
export * from './models/auto/processing_auto.js';

// Configs
export { PretrainedConfig, AutoConfig } from './configs.js';

// Additional exports
export * from './generation/streamers.js';
export * from './generation/stopping_criteria.js';
export * from './generation/logits_process.js';

export { read_audio, RawAudio } from './utils/audio.js';
export { load_image, RawImage } from './utils/image.js';
export { load_video, RawVideo, RawVideoFrame } from './utils/video.js';
export * from './utils/tensor.js';
export { softmax, log_softmax, dot, cos_sim } from './utils/maths.js';

// Expose common types used across the library for developers to access
/**
 * @typedef {import('./utils/hub.js').PretrainedModelOptions} PretrainedModelOptions
 * @typedef {import('./processing_utils.js').PretrainedProcessorOptions} PretrainedProcessorOptions
 * @typedef {import('./tokenization_utils.js').PretrainedTokenizerOptions} PretrainedTokenizerOptions
 * @typedef {import('./utils/dtypes.js').DataType} DataType
 * @typedef {import('./utils/devices.js').DeviceType} DeviceType
 * @typedef {import('./utils/core.js').ProgressCallback} ProgressCallback
 * @typedef {import('./utils/core.js').ProgressInfo} ProgressInfo
 */
