import { DEFAULT_DTYPE_SUFFIX_MAPPING, selectDtype } from '../dtypes.js';
import { selectDevice } from '../devices.js';
import { resolveExternalDataFormat, getExternalDataChunkNames } from '../model-loader.js';
import { getSessionsConfig } from '../../models/session_config.js';
import { AutoConfig } from '../../configs.js';
import { memoizePromise } from '../memoize_promise.js';
import { resolve_model_type } from './resolve_model_type.js';

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
 * @returns {Promise<PretrainedConfig>}
 */
export function get_config(
    modelId,
    { config = null, cache_dir = null, local_files_only = false, revision = 'main' } = {},
) {
    // When a pre-loaded config is provided, skip memoization — no fetch occurs
    // and there is no meaningful key to deduplicate on.
    if (config !== null) {
        return AutoConfig.from_pretrained(modelId, { config, cache_dir, local_files_only, revision });
    }
    const key = JSON.stringify([modelId, cache_dir, local_files_only, revision]);
    return memoizePromise(key, () =>
        AutoConfig.from_pretrained(modelId, { config, cache_dir, local_files_only, revision }),
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
 * @returns {Promise<string[]>} Array of file paths that will be loaded
 */
export async function get_model_files(
    modelId,
    { config = null, dtype: overrideDtype = null, device: overrideDevice = null, model_file_name = null } = {},
) {
    config = await get_config(modelId, { config });

    const files = [
        // Add config.json (always loaded)
        'config.json',
    ];
    const custom_config = config['transformers.js_config'] ?? {};

    const use_external_data_format = custom_config.use_external_data_format;
    const subfolder = 'onnx'; // Always 'onnx' as per the default in from_pretrained

    const rawDevice = overrideDevice ?? custom_config.device;
    let dtype = overrideDtype ?? custom_config.dtype;

    // Infer model type from config
    const modelType = resolve_model_type(config);

    const add_model_file = (fileName, baseName = null) => {
        baseName = baseName ?? fileName;
        const selectedDevice = selectDevice(rawDevice, fileName);
        const selectedDtype = selectDtype(dtype, fileName, selectedDevice);

        const suffix = DEFAULT_DTYPE_SUFFIX_MAPPING[selectedDtype] ?? '';
        const fullName = `${baseName}${suffix}.onnx`;
        const fullPath = subfolder ? `${subfolder}/${fullName}` : fullName;
        files.push(fullPath);

        // Check for external data files
        const num_chunks = resolveExternalDataFormat(use_external_data_format, fullName, fileName);
        for (const dataFileName of getExternalDataChunkNames(fullName, num_chunks)) {
            const dataFilePath = subfolder ? `${subfolder}/${dataFileName}` : dataFileName;
            files.push(dataFilePath);
        }
    };

    // Get session configuration from the shared source of truth
    const { sessions, optional_configs } = getSessionsConfig(modelType, config, { model_file_name });

    // Add model files based on sessions
    for (const [sessionKey, baseName] of Object.entries(sessions)) {
        add_model_file(sessionKey, baseName);
    }

    // Add optional config files
    if (optional_configs) {
        for (const configFile of Object.values(optional_configs)) {
            files.push(configFile);
        }
    }

    return files;
}
