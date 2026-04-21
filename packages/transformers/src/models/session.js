import {
    createInferenceSession,
    deviceToExecutionProviders,
    isONNXProxy,
    isONNXTensor,
    runInferenceSession,
} from '../backends/onnx.js';
import { getCacheNames } from '../configs.js';
import { DATA_TYPES, DEFAULT_DTYPE_SUFFIX_MAPPING, isWebGpuFp16Supported, selectDtype } from '../utils/dtypes.js';
import { selectDevice } from '../utils/devices.js';
import { apis } from '../env.js';
import { getCoreModelFile, getModelDataFiles } from '../utils/model-loader.js';
import { Tensor } from '../utils/tensor.js';
import { logger } from '../utils/logger.js';

/**
 * Constructs an InferenceSession using a model file located at the specified path.
 * @param {string} pretrained_model_name_or_path The path to the directory containing the model file.
 * @param {string} fileName The name of the model file.
 * @param {import('../utils/hub.js').PretrainedModelOptions} options Additional options for loading the model.
 * @param {boolean} [cache_config=false] Whether to compute cache shapes for GPU-pinned outputs.
 * @param {string} [session_name] The name of the session (used to determine cache shapes).
 * @returns {Promise<{buffer_or_path: Uint8Array|string, session_options: Object, session_config: Object}>} A Promise that resolves to the data needed to create an InferenceSession object.
 * @private
 */
async function getSession(
    pretrained_model_name_or_path,
    fileName,
    options,
    cache_config = false,
    session_name = undefined,
) {
    let custom_config = options.config?.['transformers.js_config'] ?? {};

    // If the device is not specified, we use the default (supported) execution providers.
    const selectedDevice = /** @type {import("../utils/devices.js").DeviceType} */ (
        selectDevice(options.device ?? custom_config.device, fileName, {
            warn: (msg) => logger.info(msg),
        })
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
    const selectedDtype = /** @type {import("../utils/dtypes.js").DataType} */ (
        selectDtype(options.dtype ?? custom_config.dtype, fileName, selectedDevice, {
            configDtype: custom_config.dtype,
            warn: (msg) => logger.info(msg),
        })
    );

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

    // Construct the model file suffix
    const suffix = DEFAULT_DTYPE_SUFFIX_MAPPING[selectedDtype];

    const session_options = { ...options.session_options };

    // Overwrite `executionProviders` if not specified
    session_options.executionProviders ??= executionProviders;

    // Overwrite `freeDimensionOverrides` if specified in config and not set in session options
    const free_dimension_overrides = custom_config.free_dimension_overrides;
    if (free_dimension_overrides) {
        session_options.freeDimensionOverrides ??= free_dimension_overrides;
    } else if (selectedDevice.startsWith('webnn') && !session_options.freeDimensionOverrides) {
        logger.warn(
            `WebNN does not currently support dynamic shapes and requires 'free_dimension_overrides' to be set in config.json, preferably as a field within config["transformers.js_config"]["device_config"]["${selectedDevice}"]. ` +
                `When 'free_dimension_overrides' is not set, you may experience significant performance degradation.`,
        );
    }

    const bufferOrPathPromise = getCoreModelFile(pretrained_model_name_or_path, fileName, options, suffix);

    // Handle onnx external data files
    const use_external_data_format = options.use_external_data_format ?? custom_config.use_external_data_format;
    const externalData = await getModelDataFiles(
        pretrained_model_name_or_path,
        fileName,
        suffix,
        options,
        use_external_data_format,
        session_options,
    );

    if (externalData.length > 0 && (!apis.IS_NODE_ENV || externalData.some((data) => typeof data !== 'string'))) {
        session_options.externalData = externalData;
    }

    if (cache_config && selectedDevice === 'webgpu') {
        const names = getCacheNames(options.config, {
            prefix: 'present',
            session_name,
        });
        if (names.size > 0 && !isONNXProxy()) {
            // Only set preferredOutputLocation if names are present and we aren't proxying ONNX
            /** @type {Record<string, import('onnxruntime-common').Tensor.DataLocation>} */
            const preferredOutputLocation = {};
            for (const key of names) {
                preferredOutputLocation[key] = 'gpu-buffer';
            }
            session_options.preferredOutputLocation = preferredOutputLocation;
        }
    }

    const buffer_or_path = await bufferOrPathPromise;
    const session_config = {
        dtype: selectedDtype,
        device: selectedDevice,
    };
    return { buffer_or_path, session_options, session_config };
}

/**
 * Helper function to create multiple InferenceSession objects.
 *
 * @param {string} pretrained_model_name_or_path The path to the directory containing the model file.
 * @param {Record<string, string>} names The names of the model files to load.
 * @param {import('../utils/hub.js').PretrainedModelOptions} options Additional options for loading the model.
 * @param {Record<string, true>} [cache_sessions] A map from session name to `true`, indicating which
 *   sessions should have GPU-pinned KV cache outputs.
 * @returns {Promise<Record<string, any>>} A Promise that resolves to a dictionary of InferenceSession objects.
 * @private
 */
export async function constructSessions(pretrained_model_name_or_path, names, options, cache_sessions = undefined) {
    return Object.fromEntries(
        await Promise.all(
            Object.keys(names).map(async (name) => {
                const cache_config = cache_sessions?.[name] ?? false;
                const { buffer_or_path, session_options, session_config } = await getSession(
                    pretrained_model_name_or_path,
                    names[name],
                    options,
                    cache_config,
                    name,
                );
                const session = await createInferenceSession(buffer_or_path, session_options, session_config);
                return [name, session];
            }),
        ),
    );
}

/**
 * Replaces ONNX Tensor objects with custom Tensor objects to support additional functions.
 * @param {Object} obj The object to replace tensor objects in.
 * @returns {Object} The object with tensor objects replaced by custom Tensor objects.
 * @private
 */
function replaceTensors(obj) {
    for (let prop in obj) {
        if (isONNXTensor(obj[prop])) {
            obj[prop] = new Tensor(obj[prop]);
        } else if (typeof obj[prop] === 'object') {
            replaceTensors(obj[prop]);
        }
    }
    return obj;
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
        const ortFeed = Object.fromEntries(
            Object.entries(checkedInputs).map(([k, v]) => {
                const tensor = /** @type {any} */ (v.ort_tensor);
                if (apis.IS_NODE_ENV) {
                    // In recent versions of Node.js, which support Float16Array, we need to convert
                    // the Float16Array to Uint16Array for ONNX Runtime to accept it.
                    if (typeof Float16Array !== 'undefined' && tensor.cpuData instanceof Float16Array) {
                        tensor.cpuData = new Uint16Array(tensor.cpuData.buffer); // reinterpret as Uint16Array
                    }
                }
                return [k, tensor];
            }),
        );

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
        logger.error(`An error occurred during model execution: "${e}".`);
        logger.error('Inputs given to model:', formatted);
        throw e;
    }
}

/**
 * Validate model inputs
 * @param {Object} session The InferenceSession object that will be run.
 * @param {Object} inputs The inputs to check.
 * @returns {Record<string, Tensor>} The checked inputs.
 * @throws {Error} If any inputs are missing.
 * @private
 */
function validateInputs(session, inputs) {
    /**
     * NOTE: Create either a shallow or deep copy based on `onnx.wasm.proxy`
     * @type {Record<string, Tensor>}
     */
    const checkedInputs = Object.create(null);
    const missingInputs = [];
    for (const inputName of session.inputNames) {
        const tensor = inputs[inputName];
        // Rare case where one of the model's input names corresponds to a built-in
        // object name (e.g., toString), which would cause a simple (!tensor) check to fail,
        // because it's not undefined but a function.
        if (!(tensor instanceof Tensor)) {
            missingInputs.push(inputName);
            continue;
        }
        // NOTE: When `env.wasm.proxy is true` the tensor is moved across the Worker
        // boundary, transferring ownership to the worker and invalidating the tensor.
        // So, in this case, we simply sacrifice a clone for it.
        checkedInputs[inputName] = isONNXProxy() ? tensor.clone() : tensor;
    }
    if (missingInputs.length > 0) {
        throw new Error(
            `An error occurred during model execution: "Missing the following inputs: ${missingInputs.join(', ')}.`,
        );
    }

    const numInputsProvided = Object.keys(inputs).length;
    const numInputsNeeded = session.inputNames.length;
    if (numInputsProvided > numInputsNeeded) {
        // No missing inputs, but too many inputs were provided.
        // Warn the user and ignore the extra inputs.
        let ignored = Object.keys(inputs).filter((inputName) => !session.inputNames.includes(inputName));
        logger.warn(
            `WARNING: Too many inputs were provided (${numInputsProvided} > ${numInputsNeeded}). The following inputs will be ignored: "${ignored.join(', ')}".`,
        );
    }

    return checkedInputs;
}
