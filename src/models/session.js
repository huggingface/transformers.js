import {
    createInferenceSession,
    deviceToExecutionProviders,
    isONNXProxy,
    runInferenceSession,
} from '../backends/onnx.js';
import { getCacheShapes } from '../configs.js';
import { getModelFile, MAX_EXTERNAL_DATA_CHUNKS } from '../utils/hub.js';
import {
    DATA_TYPES,
    DEFAULT_DEVICE_DTYPE_MAPPING,
    DEFAULT_DTYPE_SUFFIX_MAPPING,
    isWebGpuFp16Supported,
} from '../utils/dtypes.js';
import { apis, env } from '../env.js';
import { replaceTensors } from '../utils/tensor.js';
import { validateInputs } from './utils.js';

/**
 * Constructs an InferenceSession using a model file located at the specified path.
 * @param {string} pretrained_model_name_or_path The path to the directory containing the model file.
 * @param {string} fileName The name of the model file.
 * @param {import('./utils/hub.js').PretrainedModelOptions} options Additional options for loading the model.
 * @param {boolean} [is_decoder=false] Whether the model is a decoder model.
 * @returns {Promise<{buffer_or_path: Uint8Array|string, session_options: Object, session_config: Object}>} A Promise that resolves to the data needed to create an InferenceSession object.
 * @private
 */
async function getSession(pretrained_model_name_or_path, fileName, options, is_decoder = false) {
    let custom_config = options.config?.['transformers.js_config'] ?? {};

    let device = options.device ?? custom_config.device;
    if (device && typeof device !== 'string') {
        if (device.hasOwnProperty(fileName)) {
            device = device[fileName];
        } else {
            console.warn(`device not specified for "${fileName}". Using the default device.`);
            device = null;
        }
    }

    // If the device is not specified, we use the default (supported) execution providers.
    const selectedDevice = /** @type {import("./utils/devices.js").DeviceType} */ (
        device ?? (apis.IS_NODE_ENV ? 'cpu' : 'wasm')
    );

    const executionProviders = deviceToExecutionProviders(selectedDevice);

    // Update custom config with the selected device's config, if it exists
    const device_config = custom_config.device_config ?? {};
    if (device_config.hasOwnProperty(selectedDevice)) {
        custom_config = {
            ...custom_config,
            ...device_config[selectedDevice],
        };
    }

    // If options.dtype is specified, we use it to choose the suffix for the model file.
    // Otherwise, we use the default dtype for the device.
    let dtype = options.dtype ?? custom_config.dtype;
    if (typeof dtype !== 'string') {
        if (dtype && dtype.hasOwnProperty(fileName)) {
            dtype = dtype[fileName];
        } else {
            dtype = DEFAULT_DEVICE_DTYPE_MAPPING[selectedDevice] ?? DATA_TYPES.fp32;
            console.warn(
                `dtype not specified for "${fileName}". Using the default dtype (${dtype}) for this device (${selectedDevice}).`,
            );
        }
    }

    if (dtype === DATA_TYPES.auto) {
        // Try to choose the auto dtype based on the custom config
        let config_dtype = custom_config.dtype;
        if (typeof config_dtype !== 'string') {
            config_dtype = config_dtype?.[fileName];
        }

        if (config_dtype && config_dtype !== DATA_TYPES.auto && DATA_TYPES.hasOwnProperty(config_dtype)) {
            // Defined by the config, and is not "auto"
            dtype = config_dtype;
        } else {
            // Choose default dtype based on device, falling back to fp32
            dtype = DEFAULT_DEVICE_DTYPE_MAPPING[selectedDevice] ?? DATA_TYPES.fp32;
        }
    }

    const selectedDtype = /** @type {import("./utils/dtypes.js").DataType} */ (dtype);

    if (!DEFAULT_DTYPE_SUFFIX_MAPPING.hasOwnProperty(selectedDtype)) {
        throw new Error(`Invalid dtype: ${selectedDtype}. Should be one of: ${Object.keys(DATA_TYPES).join(', ')}`);
    } else if (
        selectedDevice === 'webgpu' &&
        // NOTE: Currently, we assume that the Native WebGPU EP always supports fp16. In future, we will add a check for this.
        !apis.IS_NODE_ENV &&
        selectedDtype === DATA_TYPES.fp16 &&
        !(await isWebGpuFp16Supported())
    ) {
        throw new Error(`The device (${selectedDevice}) does not support fp16.`);
    }

    // Only valid for models with a decoder
    const kv_cache_dtype_config = custom_config.kv_cache_dtype;
    const kv_cache_dtype = kv_cache_dtype_config
        ? typeof kv_cache_dtype_config === 'string'
            ? kv_cache_dtype_config
            : (kv_cache_dtype_config[selectedDtype] ?? 'float32')
        : undefined;

    if (kv_cache_dtype && !['float32', 'float16'].includes(kv_cache_dtype)) {
        throw new Error(`Invalid kv_cache_dtype: ${kv_cache_dtype}. Should be one of: float32, float16`);
    }

    const session_config = {
        dtype: selectedDtype,
        kv_cache_dtype,
        device: selectedDevice,
    };

    // Construct the model file name
    const suffix = DEFAULT_DTYPE_SUFFIX_MAPPING[selectedDtype];
    const baseName = `${fileName}${suffix}.onnx`;
    const modelFileName = `${options.subfolder ?? ''}/${baseName}`;

    const session_options = { ...options.session_options };

    // Overwrite `executionProviders` if not specified
    session_options.executionProviders ??= executionProviders;

    // Overwrite `freeDimensionOverrides` if specified in config and not set in session options
    const free_dimension_overrides = custom_config.free_dimension_overrides;
    if (free_dimension_overrides) {
        session_options.freeDimensionOverrides ??= free_dimension_overrides;
    } else if (selectedDevice.startsWith('webnn') && !session_options.freeDimensionOverrides) {
        console.warn(
            `WebNN does not currently support dynamic shapes and requires 'free_dimension_overrides' to be set in config.json, preferably as a field within config["transformers.js_config"]["device_config"]["${selectedDevice}"]. ` +
                `When 'free_dimension_overrides' is not set, you may experience significant performance degradation.`,
        );
    }

    const return_path = apis.IS_NODE_ENV && env.useFSCache;
    const bufferOrPathPromise = getModelFile(pretrained_model_name_or_path, modelFileName, true, options, return_path);

    // Handle onnx external data files
    const use_external_data_format = options.use_external_data_format ?? custom_config.use_external_data_format;
    /** @type {Promise<string|{path: string, data: Uint8Array}>[]} */
    let externalDataPromises = [];
    if (use_external_data_format) {
        let external_data_format;
        if (typeof use_external_data_format === 'object') {
            if (use_external_data_format.hasOwnProperty(baseName)) {
                external_data_format = use_external_data_format[baseName];
            } else if (use_external_data_format.hasOwnProperty(fileName)) {
                external_data_format = use_external_data_format[fileName];
            } else {
                external_data_format = false;
            }
        } else {
            external_data_format = use_external_data_format;
        }

        const num_chunks = +external_data_format; // (false=0, true=1, number remains the same)
        if (num_chunks > MAX_EXTERNAL_DATA_CHUNKS) {
            throw new Error(
                `The number of external data chunks (${num_chunks}) exceeds the maximum allowed value (${MAX_EXTERNAL_DATA_CHUNKS}).`,
            );
        }
        for (let i = 0; i < num_chunks; ++i) {
            const path = `${baseName}_data${i === 0 ? '' : '_' + i}`;
            const fullPath = `${options.subfolder ?? ''}/${path}`;
            externalDataPromises.push(
                new Promise(async (resolve, reject) => {
                    const data = await getModelFile(
                        pretrained_model_name_or_path,
                        fullPath,
                        true,
                        options,
                        return_path,
                    );
                    resolve(data instanceof Uint8Array ? { path, data } : path);
                }),
            );
        }
    } else if (session_options.externalData !== undefined) {
        externalDataPromises = session_options.externalData.map(async (ext) => {
            // if the external data is a string, fetch the file and replace the string with its content
            // @ts-expect-error TS2339
            if (typeof ext.data === 'string') {
                // @ts-expect-error TS2339
                const ext_buffer = await getModelFile(pretrained_model_name_or_path, ext.data, true, options);
                // @ts-expect-error TS2698
                return { ...ext, data: ext_buffer };
            }
            return ext;
        });
    }

    if (externalDataPromises.length > 0) {
        const externalData = await Promise.all(externalDataPromises);
        if (!apis.IS_NODE_ENV) {
            session_options.externalData = externalData;
        }
    }

    if (is_decoder && selectedDevice === 'webgpu' && kv_cache_dtype_config !== false) {
        const shapes = getCacheShapes(options.config, {
            prefix: 'present',
        });
        if (Object.keys(shapes).length > 0 && !isONNXProxy()) {
            // Only set preferredOutputLocation if shapes are present and we aren't proxying ONNX
            /** @type {Record<string, import('onnxruntime-common').Tensor.DataLocation>} */
            const preferredOutputLocation = {};
            for (const key in shapes) {
                preferredOutputLocation[key] = 'gpu-buffer';
            }
            session_options.preferredOutputLocation = preferredOutputLocation;
        }
    }

    const buffer_or_path = await bufferOrPathPromise;

    return { buffer_or_path, session_options, session_config };
}

/**
 * Helper function to create multiple InferenceSession objects.
 *
 * @param {string} pretrained_model_name_or_path The path to the directory containing the model file.
 * @param {Record<string, string>} names The names of the model files to load.
 * @param {import('./utils/hub.js').PretrainedModelOptions} options Additional options for loading the model.
 * @param {string} [decoder_name] The name of the decoder model, if any.
 * @returns {Promise<Record<string, any>>} A Promise that resolves to a dictionary of InferenceSession objects.
 * @private
 */
export async function constructSessions(pretrained_model_name_or_path, names, options, decoder_name = undefined) {
    return Object.fromEntries(
        await Promise.all(
            Object.keys(names).map(async (name) => {
                const { buffer_or_path, session_options, session_config } = await getSession(
                    pretrained_model_name_or_path,
                    names[name],
                    options,
                    name === decoder_name,
                );
                const session = await createInferenceSession(buffer_or_path, session_options, session_config);
                return [name, session];
            }),
        ),
    );
}

/**
 * Executes an InferenceSession using the specified inputs.
 * NOTE: `inputs` must contain at least the input names of the model.
 *  - If additional inputs are passed, they will be ignored.
 *  - If inputs are missing, an error will be thrown.
 *
 * @param {Object} session The InferenceSession object to run.
 * @param {Object} inputs An object that maps input names to input tensors.
 * @returns {Promise<Object>} A Promise that resolves to an object that maps output names to output tensors.
 * @private
 */
export async function sessionRun(session, inputs) {
    const checkedInputs = validateInputs(session, inputs);
    try {
        // pass the original ort tensor
        const ortFeed = Object.fromEntries(Object.entries(checkedInputs).map(([k, v]) => [k, v.ort_tensor]));
        const output = await runInferenceSession(session, ortFeed);
        return replaceTensors(output);
    } catch (e) {
        // Error messages can be long (nested) and uninformative. For this reason,
        // we apply minor formatting to show the most important information
        const formatted = Object.fromEntries(
            Object.entries(checkedInputs).map(([k, tensor]) => {
                // Extract these properties from the underlying ORT tensor
                const unpacked = {
                    type: tensor.type,
                    dims: tensor.dims,
                    location: tensor.location,
                };
                if (unpacked.location !== 'gpu-buffer') {
                    // Only return the data if it's not a GPU buffer
                    unpacked.data = tensor.data;
                }
                return [k, unpacked];
            }),
        );

        // This usually occurs when the inputs are of the wrong type.
        console.error(`An error occurred during model execution: "${e}".`);
        console.error('Inputs given to model:', formatted);
        throw e;
    }
}
