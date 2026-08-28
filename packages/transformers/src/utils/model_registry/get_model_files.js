import { AutoConfig } from '../../configs.js';
import { makePretrainedOptionsKey } from '../hub/utils.js';
import { memoizePromise } from '../memoize_promise.js';
import { resolve_model_type } from './resolve_model_type.js';
import { getModelRegistryInferenceProvider } from '../../backends/model_registry.js';
import { withInferenceBackendHostOptions } from '../../backends/inference.js';

/**
 * @typedef {import('../../configs.js').PretrainedConfig} PretrainedConfig
 */

/**
 * Returns a memoized AutoConfig for the given model ID and options.
 * If the same model ID and options have been requested before — even while
 * the first request is still in-flight — the cached promise is returned
 * so that config.json is only fetched once.
 * When a pre-loaded `config` object is supplied the result is not memoized,
 * since the caller already has the config and no network operation is performed.
 *
 * @param {string} modelId The model id (e.g., "onnx-community/granite-4.0-350m-ONNX-web")
 * @param {Object} [options]
 * @param {PretrainedConfig|null} [options.config=null] Pre-loaded config; skips fetching if provided.
 * @param {string|null} [options.cache_dir=null] Custom local cache directory.
 * @param {boolean} [options.local_files_only=false] Never hit the network if true.
 * @param {string} [options.revision='main'] Git branch, tag, or commit SHA.
 * @param {string|null} [options.subfolder=null] Optional directory containing shared assets.
 * @param {AbortSignal} [options.signal] Cancellation signal.
 * @returns {Promise<PretrainedConfig>}
 */
export function get_config(
    modelId,
    { config = null, cache_dir = null, local_files_only = false, revision = 'main', subfolder = null, signal = undefined } = {},
) {
    // When a pre-loaded config is provided, skip memoization — no fetch occurs
    // and there is no meaningful key to deduplicate on.
    if (config !== null) {
        return AutoConfig.from_pretrained(modelId, { config, cache_dir, local_files_only, revision, subfolder, signal });
    }
    if (signal) {
        return AutoConfig.from_pretrained(modelId, { config, cache_dir, local_files_only, revision, subfolder, signal });
    }
    const key = makePretrainedOptionsKey(modelId, { cache_dir, local_files_only, revision, subfolder });
    return memoizePromise(key, () =>
        AutoConfig.from_pretrained(modelId, { config, cache_dir, local_files_only, revision, subfolder, signal }),
    );
}

/**
 * Returns the list of files that will be loaded for a model based on its configuration.
 *
 * This function reads configuration from the model's config.json on the hub.
 * If dtype/device are not specified in the config, you can provide them to match
 * what the pipeline will actually use.
 *
 * @param {string} modelId The model id (e.g., "onnx-community/granite-4.0-350m-ONNX-web")
 * @param {Object} [options] Optional parameters
 * @param {import('../../configs.js').PretrainedConfig} [options.config=null] Pre-loaded model config (optional, will be fetched if not provided)
 * @param {import('../dtypes.js').DataType|Record<string, import('../dtypes.js').DataType>} [options.dtype=null] Override dtype (use this if passing dtype to pipeline)
 * @param {import('../devices.js').DeviceType|Record<string, import('../devices.js').DeviceType>} [options.device=null] Override device (use this if passing device to pipeline)
 * @param {string} [options.model_file_name=null] Override the model file name (excluding .onnx suffix).
 * @param {string|null} [options.cache_dir=null] Custom cache directory.
 * @param {boolean} [options.local_files_only=false] Never hit the network if true.
 * @param {string} [options.revision='main'] Model revision.
 * @param {string|null} [options.subfolder=null] Optional directory containing shared assets.
 * @param {AbortSignal} [options.signal] Cancellation signal.
 * @param {string|null} [options.task=null] Pipeline task requesting the artifacts.
 * @param {boolean|number|Record<string, boolean|number>} [options.use_external_data_format=null] ONNX external-data configuration.
 * @param {import('../../backends/model_registry.js').ModelRegistryInferenceProvider|null} [options.inferenceProvider=null] Artifact metadata provider.
 * @returns {Promise<string[]>} Array of file paths that will be loaded
 */
export async function get_model_files(
    modelId,
    {
        config = null,
        dtype: overrideDtype = null,
        device: overrideDevice = null,
        model_file_name = null,
        cache_dir = null,
        local_files_only = false,
        revision = 'main',
        subfolder = null,
        signal = undefined,
        use_external_data_format = null,
        task = null,
        inferenceProvider = null,
    } = {},
) {
    config = await get_config(modelId, { config, cache_dir, local_files_only, revision, subfolder, signal });

    // Infer model type from config
    const modelType = resolve_model_type(config);
    const provider = await getModelRegistryInferenceProvider(inferenceProvider);
    return [
        ...(await provider.listModelArtifacts(withInferenceBackendHostOptions({
            modelId,
            task,
            modelType,
            config,
            model_file_name,
            dtype: overrideDtype,
            device: overrideDevice,
            cache_dir,
            local_files_only,
            revision,
            subfolder,
            signal,
            use_external_data_format,
        }))),
    ];
}
