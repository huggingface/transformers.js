import { getSessionsConfig } from '../../models/session_config.js';
import { get_file_metadata } from './get_file_metadata.js';
import { get_config } from './get_model_files.js';
import { resolve_model_type } from './resolve_model_type.js';
import { OnnxInferenceProvider } from '../../backends/default.js';

/**
 * @typedef {import('../../configs.js').PretrainedConfig} PretrainedConfig
 */

/**
 * The dtypes to probe for availability (excludes 'auto' which is not a concrete dtype).
 * @type {string[]}
 */
/**
 * Detects which quantization levels (dtypes) are available for a model
 * by checking which ONNX files exist on the hub or locally.
 *
 * A dtype is considered available if *all* required model session files
 * exist for that dtype. For example, a Seq2Seq model needs both an encoder
 * and decoder file — the dtype is only listed if both are present.
 *
 * @param {string} modelId The model id (e.g., "onnx-community/all-MiniLM-L6-v2-ONNX")
 * @param {Object} [options] Optional parameters
 * @param {PretrainedConfig} [options.config=null] Pre-loaded model config (optional, will be fetched if not provided)
 * @param {string} [options.model_file_name=null] Override the model file name (excluding .onnx suffix)
 * @param {string} [options.revision='main'] Model revision
 * @param {string} [options.cache_dir=null] Custom cache directory
 * @param {boolean} [options.local_files_only=false] Only check local files
 * @returns {Promise<string[]>} Array of available dtype strings (e.g., ['fp32', 'fp16', 'q4', 'q8'])
 */
export async function get_available_dtypes(
    modelId,
    { config = null, model_file_name = null, revision = 'main', cache_dir = null, local_files_only = false } = {},
) {
    config = await get_config(modelId, { config, cache_dir, local_files_only, revision });

    const modelType = resolve_model_type(config);
    const { sessions } = getSessionsConfig(modelType, config, { model_file_name });
    const metadataOptions = { revision, cache_dir, local_files_only };
    return OnnxInferenceProvider.getAvailableDtypes({
        modelId,
        sessions,
        getFileMetadata: get_file_metadata,
        metadataOptions,
    });
}
