/**
 * @file Public API of `@huggingface/transformers`. Everything re-exported from
 * this file is considered stable — other imports are internal and may change.
 *
 * **Start here**
 * - [`pipeline()`](./pipelines.md#pipeline) — the one-call entry point for every task.
 * - [Environment](./env.md) — `env` fields and `LogLevel` enum.
 *
 * **Model loading**
 * - [Pipelines](./pipelines.md) — task-specific pipeline classes (`TextGenerationPipeline`, etc.).
 * - [Models](./models.md) — `AutoModel*` classes (one per task).
 * - [Tokenizers](./tokenizers.md) — `AutoTokenizer`, chat templates, `Message`.
 * - [Processors](./processors.md) — `AutoProcessor`, `AutoImageProcessor`, `AutoFeatureExtractor`.
 * - [Configs](./configs.md) — `AutoConfig` / `PretrainedConfig`.
 *
 * **Generation**
 * - [Generation config](./generation/configuration_utils.md) — sampling and beam-search parameters.
 * - [Generation parameters](./generation/parameters.md) — full shape of `generate()` arguments.
 * - [Logits processors](./generation/logits_process.md) — modify next-token probabilities.
 * - [Stopping criteria](./generation/stopping_criteria.md) — control when generation halts.
 * - [Streamers](./generation/streamers.md) — receive tokens as they're produced.
 *
 * **Data types and I/O**
 * - [Tensors](./utils/tensor.md) — `Tensor`, shape ops, math, I/O.
 * - [Images](./utils/image.md) — `RawImage`, `load_image()`.
 * - [Audio](./utils/audio.md) — `RawAudio`, `load_audio()`.
 * - [Video](./utils/video.md) — `RawVideo`, `RawVideoFrame`, `load_video()` _(experimental)_.
 *
 * **Utilities**
 * - [Hub options](./utils/hub.md) — shared `from_pretrained()` option shapes.
 * - [Maths](./utils/maths.md) — `softmax`, `cos_sim`, typed-array helpers.
 * - [Model registry](./utils/model_registry.md) — inspect or clear the model cache.
 * - [Random](./utils/random.md) — seedable MT19937 PRNG matching Python's `random`.
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
export { GenerationConfig } from './generation/configuration_utils.js';

export { load_audio, read_audio, RawAudio } from './utils/audio.js';
export { load_image, RawImage } from './utils/image.js';
export { load_video, RawVideo, RawVideoFrame } from './utils/video.js';
export * from './utils/tensor.js';
export { softmax, log_softmax, dot, cos_sim } from './utils/maths.js';
export { random } from './utils/random.js';

export { DynamicCache } from './cache_utils.js';

// Cache and file management
export { ModelRegistry } from './utils/model_registry/ModelRegistry.js';

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
 */
