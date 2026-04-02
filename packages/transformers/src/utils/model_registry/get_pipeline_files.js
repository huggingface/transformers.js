import { get_files } from './get_files.js';
import { get_config } from './get_model_files.js';
import { resolve_model_type } from './resolve_model_type.js';
import { getTextOnlySessions } from '../../models/session_config.js';
import { SUPPORTED_TASKS, TASK_ALIASES } from '../../pipelines/index.js';
import { MODEL_TYPE_MAPPING } from '../../models/modeling_utils.js';

/**
 * Determines the effective model type for a given pipeline task and model config by simulating
 * the auto class resolution that `pipeline()` itself would perform.
 *
 * `resolve_model_type` uses `config.architectures` (e.g. `["Qwen3ForCausalLM"]`) which maps to
 * `DecoderOnly` and includes `optional_configs` like `generation_config.json`. But a task like
 * `feature-extraction` uses `AutoModel`, which iterates its `MODEL_CLASS_MAPPINGS` in order and
 * finds the *base* model class (e.g. `Qwen3Model` → `DecoderOnlyWithoutHead`) — a type that has
 * no `optional_configs`. This function replicates that lookup so the file list matches what the
 * pipeline actually loads.
 *
 * Cross-architecture detection is also replicated: when a `ForCausalLM` class loads a model
 * whose native architecture ends in `ForConditionalGeneration`, the native type is used instead
 * (matching the logic in `resolveTypeConfig` in `modeling_utils.js`).
 *
 * @param {Object} taskConfig - The pipeline task config from SUPPORTED_TASKS.
 * @param {import('../../configs.js').PretrainedConfig} config - The model config.
 * @returns {number|null} The resolved MODEL_TYPES value, or null if unresolvable (fall back to resolve_model_type).
 */
function resolveEffectiveModelType(taskConfig, config) {
    const { model_type } = config;
    if (!model_type) return null;

    // @ts-ignore - architectures is assigned dynamically via Object.assign in PretrainedConfig
    const nativeArch = config.architectures?.[0];
    // @ts-ignore - model is a dynamic property on SUPPORTED_TASKS entries
    const autoClasses = Array.isArray(taskConfig.model) ? taskConfig.model : [taskConfig.model];

    for (const autoClass of autoClasses) {
        if (!autoClass?.MODEL_CLASS_MAPPINGS) continue;

        for (const mapping of autoClass.MODEL_CLASS_MAPPINGS) {
            const className = mapping.get(model_type);
            if (className === undefined) continue;

            let type = MODEL_TYPE_MAPPING.get(className);
            if (type === undefined) continue;

            // Replicate cross-architecture detection from resolveTypeConfig in modeling_utils.js:
            // A ForCausalLM class loading a ForConditionalGeneration model uses the native type.
            if (
                nativeArch &&
                nativeArch !== className &&
                className.endsWith('ForCausalLM') &&
                nativeArch.endsWith('ForConditionalGeneration')
            ) {
                const nativeType = MODEL_TYPE_MAPPING.get(nativeArch);
                if (nativeType !== undefined) type = nativeType;
            }

            return type;
        }
    }

    return null;
}

/**
 * Get all files needed for a specific pipeline task.
 * Automatically detects which components (tokenizer, processor) are needed by checking
 * whether the model has the corresponding files (tokenizer_config.json, preprocessor_config.json).
 *
 * @param {string} task - The pipeline task (e.g., "text-generation", "image-classification")
 * @param {string} modelId - The model id (e.g., "Xenova/bert-base-uncased")
 * @param {Object} [options] - Optional parameters
 * @param {import('../../configs.js').PretrainedConfig} [options.config=null] - Pre-loaded config
 * @param {import('../dtypes.js').DataType|Record<string, import('../dtypes.js').DataType>} [options.dtype=null] - Override dtype
 * @param {import('../devices.js').DeviceType|Record<string, import('../devices.js').DeviceType>} [options.device=null] - Override device
 * @param {string} [options.model_file_name=null] - Override the model file name (excluding .onnx suffix)
 * @returns {Promise<string[]>} Array of file paths that will be loaded
 * @throws {Error} If the task is not supported
 */
export async function get_pipeline_files(task, modelId, options = {}) {
    // Apply task aliases
    task = TASK_ALIASES[task] ?? task;

    // Validate that the task is supported
    const taskConfig = SUPPORTED_TASKS[task];
    if (!taskConfig) {
        throw new Error(
            `Unsupported pipeline task: ${task}. Must be one of [${Object.keys(SUPPORTED_TASKS).join(', ')}]`,
        );
    }

    // Use the task type to determine which components to auto-detect:
    //  - 'text' tasks: always check tokenizer, skip processor (text models rarely have one)
    //  - 'audio'/'image' tasks: skip tokenizer, always check processor
    //  - 'multimodal' tasks: check both
    const { type } = taskConfig;
    const include_tokenizer = type !== 'audio' && type !== 'image';
    const include_processor = type !== 'text';

    // Resolve the config once up front so we can derive the effective model type.
    // get_config is memoized, so this doesn't add a redundant network fetch.
    const config = await get_config(modelId, options);

    // Determine which model type the pipeline would actually load (based on its auto class), not
    // just what config.architectures says. This prevents including optional config files (e.g.
    // generation_config.json) that the pipeline never fetches for base-model tasks like
    // feature-extraction.
    const model_type_override = resolveEffectiveModelType(taskConfig, config);

    const files = await get_files(modelId, {
        ...options,
        config,
        include_tokenizer,
        include_processor,
        model_type_override,
    });

    // When loading multimodal models via the text-generation pipeline,
    // only load the sessions needed for text generation (embed_tokens, decoder_model_merged)
    if (task === 'text-generation') {
        const modelType = model_type_override ?? resolve_model_type(config);
        const textOnlySessions = getTextOnlySessions(modelType);

        if (textOnlySessions) {
            const allowedPrefixes = Object.values(textOnlySessions).map((s) => `onnx/${s}`);
            return files.filter((f) => !f.startsWith('onnx/') || allowedPrefixes.some((p) => f.startsWith(p)));
        }
    }

    return files;
}
