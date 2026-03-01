import { get_files } from './get_files.js';
import { SUPPORTED_TASKS, TASK_ALIASES } from '../../pipelines/index.js';

/**
 * Extract component requirements from SUPPORTED_TASKS
 * @private
 * @param {string} task
 * @returns {{tokenizer: boolean, processor: boolean}}
 */
function get_task_components(task) {
    const taskConfig = SUPPORTED_TASKS[task];
    if (!taskConfig) {
        return null;
    }
    return {
        tokenizer: !!taskConfig.tokenizer,
        processor: !!taskConfig.processor,
    };
}

/**
 * Get all files needed for a specific pipeline task.
 * Automatically determines which components (tokenizer, processor) are needed based on the task.
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

    // Get component requirements for this task from SUPPORTED_TASKS
    const components = get_task_components(task);
    if (!components) {
        throw new Error(
            `Unsupported pipeline task: ${task}. Must be one of [${Object.keys(SUPPORTED_TASKS).join(', ')}]`,
        );
    }

    // Get files with appropriate component flags
    return get_files(modelId, {
        ...options,
        include_tokenizer: components.tokenizer,
        include_processor: components.processor,
    });
}
