import { get_files } from './get_files.js';
import { get_config } from './get_model_files.js';
import { resolve_model_type } from './resolve_model_type.js';
import { MODEL_TYPES } from '../../models/modeling_utils.js';
import { SUPPORTED_TASKS, TASK_ALIASES } from '../../pipelines/index.js';

/**
 * Maps multimodal model types to the encoder session names that should be
 * excluded when the model is used for a text-only pipeline task.
 */
const MULTIMODAL_ENCODER_SESSIONS = {
    [MODEL_TYPES.ImageTextToText]: ['vision_encoder'],
    [MODEL_TYPES.AudioTextToText]: ['audio_encoder'],
    [MODEL_TYPES.ImageAudioTextToText]: ['vision_encoder', 'audio_encoder'],
    [MODEL_TYPES.VoxtralRealtime]: ['audio_encoder'],
};

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

    let files = await get_files(modelId, {
        ...options,
        include_tokenizer,
        include_processor,
    });

    // For text-only tasks on multimodal models, exclude encoder files
    // (e.g., vision_encoder, audio_encoder) that are not needed for text generation.
    if (type === 'text') {
        const config = await get_config(modelId, options);
        const modelType = resolve_model_type(config);
        const excludeSessions = MULTIMODAL_ENCODER_SESSIONS[modelType];
        if (excludeSessions) {
            files = files.filter((f) => !excludeSessions.some((s) => f.includes(s)));
        }
    }

    return files;
}
