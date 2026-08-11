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
export { env, LogLevel } from './env.js';

// Pipelines
export * from './pipelines.js';

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
export { GenerationController, createGenerationController } from './generation/controller.js';

export { load_audio, read_audio, RawAudio } from './utils/audio.js';
export { load_image, RawImage } from './utils/image.js';
export { load_video, RawVideo, RawVideoFrame } from './utils/video.js';
export * from './utils/tensor.js';
export { softmax, log_softmax, dot, cos_sim } from './utils/maths.js';
export { random } from './utils/random.js';
export { logger } from './utils/logger.js';

export { DynamicCache } from './cache_utils.js';

// Cache and file management
export { ModelRegistry } from './utils/model_registry/ModelRegistry.js';

// Inference backends
export { getModelId, isInferenceBackend } from './backends/inference.js';

// Expose common types used across the library for developers to access
/**
 * @typedef {import('./utils/hub.js').PretrainedModelOptions} PretrainedModelOptions
 * @typedef {import('./processing_utils.js').PretrainedProcessorOptions} PretrainedProcessorOptions
 * @typedef {import('./tokenization_utils.js').Message} Message
 * @typedef {import('./tokenization_utils.js').PretrainedTokenizerOptions} PretrainedTokenizerOptions
 * @typedef {import('./utils/dtypes.js').DataType} DataType
 * @typedef {import('./utils/devices.js').DeviceType} DeviceType
 * @typedef {import('./utils/core.js').ProgressCallback} ProgressCallback
 * @typedef {import('./utils/core.js').ProgressInfo} ProgressInfo
 * @typedef {import('./backends/inference.js').InferenceBackend} InferenceBackend
 * @typedef {import('./backends/inference.js').InferenceBackendChatTemplate} InferenceBackendChatTemplate
 * @typedef {import('./backends/inference.js').InferenceBackendLoadOptions} InferenceBackendLoadOptions
 * @typedef {import('./backends/inference.js').InferenceModel} InferenceModel
 * @typedef {import('./backends/inference.js').InferenceModelCapabilities} InferenceModelCapabilities
 * @typedef {import('./backends/inference.js').StaticBackendCapabilities} StaticBackendCapabilities
 * @typedef {import('./backends/inference.js').ForwardCapabilitiesV1} ForwardCapabilitiesV1
 * @typedef {import('./generation/runtime.js').CausalGenerationCapabilitiesV1} CausalGenerationCapabilitiesV1
 * @typedef {import('./generation/runtime.js').GenerationCapabilitiesV1} GenerationCapabilitiesV1
 * @typedef {import('./generation/runtime.js').AutoregressiveSessionV1} AutoregressiveSessionV1
 * @typedef {import('./generation/runtime.js').PlanAutoregressiveSessionV1} PlanAutoregressiveSessionV1
 * @typedef {import('./generation/runtime.js').PullAutoregressiveSessionV1} PullAutoregressiveSessionV1
 * @typedef {import('./generation/runtime.js').SessionConcurrencyCapabilities} SessionConcurrencyCapabilities
 * @typedef {import('./generation/runtime.js').AutoregressiveSessionOptionsV1} AutoregressiveSessionOptionsV1
 * @typedef {import('./generation/runtime.js').AutoregressivePrefillInputsV1} AutoregressivePrefillInputsV1
 * @typedef {import('./generation/runtime.js').AutoregressiveDecodeInputsV1} AutoregressiveDecodeInputsV1
 * @typedef {import('./generation/runtime.js').RuntimeGenerationPlanV1} RuntimeGenerationPlanV1
 * @typedef {import('./generation/runtime.js').RuntimeTokenDecisionV1} RuntimeTokenDecisionV1
 * @typedef {import('./generation/runtime.js').RuntimeTokenBatchV1} RuntimeTokenBatchV1
 * @typedef {import('./generation/runtime.js').RuntimeAttentionMaskV1} RuntimeAttentionMaskV1
 * @typedef {import('./generation/runtime.js').LogitsLeaseV1} LogitsLeaseV1
 * @typedef {import('./backends/artifacts.js').InferenceArtifactProvider} InferenceArtifactProvider
 * @typedef {import('./backends/artifacts.js').RandomAccessByteSource} RandomAccessByteSource
 * @typedef {import('./backends/artifacts.js').ArtifactProgressEvent} ArtifactProgressEvent
 */
