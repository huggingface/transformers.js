import { get_file_metadata } from './get_file_metadata.js';
import { get_config } from './get_model_files.js';
import { resolve_model_type } from './resolve_model_type.js';
import { getModelRegistryInferenceProvider } from '../../backends/model_registry.js';
import { withInferenceBackendHostOptions } from '../../backends/inference.js';

/**
 * @typedef {import('../../configs.js').PretrainedConfig} PretrainedConfig
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
 * @param {string|null} [options.subfolder=null] Optional directory containing shared assets
 * @param {AbortSignal} [options.signal] Cancellation signal
 * @param {import('../../backends/model_registry.js').ModelRegistryInferenceProvider|null} [options.inferenceProvider=null] Artifact metadata provider
 * @returns {Promise<string[]>} Array of available dtype strings (e.g., ['fp32', 'fp16', 'q4', 'q8'])
 */
export async function get_available_dtypes(
    modelId,
    {
        config = null,
        model_file_name = null,
        revision = 'main',
        cache_dir = null,
        local_files_only = false,
        subfolder = null,
        signal = undefined,
        inferenceProvider = null,
    } = {},
) {
    config = await get_config(modelId, { config, cache_dir, local_files_only, revision, subfolder, signal });

    const modelType = resolve_model_type(config);
    const metadataOptions = { revision, cache_dir, local_files_only, subfolder, signal };
    const provider = await getModelRegistryInferenceProvider(inferenceProvider);
    if (!provider.getAvailableDtypes) {
        throw new Error('The inference backend does not support dtype discovery.');
    }
    return provider.getAvailableDtypes(withInferenceBackendHostOptions({
        modelId,
        modelType,
        config,
        model_file_name,
        getFileMetadata: get_file_metadata,
        metadataOptions,
    }));
}
