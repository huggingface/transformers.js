import {
    createInferenceSession,
    deviceToExecutionProviders,
    isONNXProxy,
    isONNXTensor,
    runInferenceSession,
    Tensor as OrtTensor,
} from './runtime.js';
import { getOnnxProviderHost } from './host.js';

const DATA_TYPES = Object.freeze({
    auto: 'auto',
    fp32: 'fp32',
    fp16: 'fp16',
    q8: 'q8',
    int8: 'int8',
    uint8: 'uint8',
    q4: 'q4',
    bnb4: 'bnb4',
    q4f16: 'q4f16',
    q2: 'q2',
    q2f16: 'q2f16',
    q1: 'q1',
    q1f16: 'q1f16',
});

const DEFAULT_DTYPE_SUFFIX_MAPPING: Record<string, string> = Object.freeze({
    fp32: '',
    fp16: '_fp16',
    int8: '_int8',
    uint8: '_uint8',
    q8: '_quantized',
    q4: '_q4',
    q2: '_q2',
    q1: '_q1',
    q4f16: '_q4f16',
    q2f16: '_q2f16',
    q1f16: '_q1f16',
    bnb4: '_bnb4',
});

const { apis, logger } = getOnnxProviderHost();

function selectDevice(value: any, fileName: string, { warn }: any = {}): string {
    const fallback = apis.IS_NODE_ENV ? 'cpu' : 'wasm';
    if (!value) return fallback;
    if (typeof value === 'string') return value;
    if (Object.hasOwn(value, fileName)) return value[fileName];
    warn?.(`device not specified for "${fileName}". Using the default device (${fallback}).`);
    return fallback;
}

function selectDtype(value: any, fileName: string, device: string, { configDtype = null, warn }: any = {}): string {
    let resolved = value;
    let needsWarn = false;
    if (value && typeof value !== 'string') {
        resolved = Object.hasOwn(value, fileName) ? value[fileName] : null;
        needsWarn = resolved === null;
    }
    if (resolved === 'auto') {
        const fallback = typeof configDtype === 'string' ? configDtype : configDtype?.[fileName];
        if (fallback && fallback !== 'auto' && Object.hasOwn(DATA_TYPES, fallback)) return fallback;
    }
    const result =
        resolved && resolved !== 'auto' && Object.hasOwn(DATA_TYPES, resolved)
            ? resolved
            : device === 'wasm'
              ? 'q8'
              : 'fp32';
    if (needsWarn)
        warn?.(
            `dtype not specified for "${fileName}". Using the default dtype (${result}) for this device (${device}).`,
        );
    return result;
}

let fp16Supported: boolean | undefined;
async function isWebGpuFp16Supported(): Promise<boolean> {
    if (fp16Supported === undefined) {
        try {
            const adapter = await navigator.gpu.requestAdapter();
            fp16Supported = !!adapter?.features.has('shader-f16');
        } catch {
            fp16Supported = false;
        }
    }
    return fp16Supported;
}

/**
 * ONNX Runtime adapter used for string model IDs.
 */
export class OnnxInferenceProvider {
    /**
     * @param {string} modelId
     * @param {typeof import('../../models/modeling_utils.js').PreTrainedModel} [modelClass]
     */
    static from_modelId(modelId: string): OnnxInferenceProvider {
        return new OnnxInferenceProvider(modelId);
    }

    static listModelArtifacts({
        sessions,
        optionalConfigs,
        config,
        dtype: overrideDtype = null,
        device: overrideDevice = null,
    }: any): string[] {
        const files = ['config.json'];
        const customConfig = config?.['transformers.js_config'] ?? {};
        const rawDevice = overrideDevice ?? customConfig.device;
        const dtype = overrideDtype ?? customConfig.dtype;
        for (const [sessionName, baseName] of Object.entries(sessions) as [string, string][]) {
            const device = selectDevice(rawDevice, sessionName);
            const selectedDtype = selectDtype(dtype, sessionName, device);
            const suffix = DEFAULT_DTYPE_SUFFIX_MAPPING[selectedDtype] ?? '';
            const fullName = `${baseName}${suffix}.onnx`;
            files.push(`onnx/${fullName}`);

            const externalConfig = customConfig.use_external_data_format;
            const count =
                typeof externalConfig === 'object' && externalConfig !== null
                    ? +(externalConfig[fullName] ?? externalConfig[sessionName] ?? 0)
                    : +(externalConfig ?? 0);
            files.push(...externalDataChunkNames(fullName, count).map((name) => `onnx/${name}`));
        }
        if (optionalConfigs) files.push(...(Object.values(optionalConfigs) as string[]));
        return files;
    }

    static async getAvailableDtypes({ modelId, sessions, getFileMetadata, metadataOptions }: any): Promise<string[]> {
        const results = await Promise.all(
            Object.entries(DEFAULT_DTYPE_SUFFIX_MAPPING).map(async ([dtype, suffix]) => ({
                dtype,
                available: (
                    await Promise.all(
                        (Object.values(sessions) as string[]).map(
                            async (baseName) =>
                                (await getFileMetadata(modelId, `onnx/${baseName}${suffix}.onnx`, metadataOptions))
                                    .exists,
                        ),
                    )
                ).every(Boolean),
            })),
        );
        return results.filter((result) => result.available).map((result) => result.dtype);
    }

    static filterModelArtifacts(files: string[], sessions: Record<string, string>): string[] {
        const allowedPrefixes = Object.values(sessions).map((name) => `onnx/${name}`);
        return files.filter(
            (file) => !file.startsWith('onnx/') || allowedPrefixes.some((prefix) => file.startsWith(prefix)),
        );
    }

    readonly providerType = 'onnx';
    modelClass: any;

    constructor(
        public readonly modelId: string,
        modelClass: any = undefined,
    ) {
        this.modelClass = modelClass;
    }

    /**
     * Load a Transformers.js model class with this backend.
     *
     * @param {import('../../utils/hub.js').PretrainedModelOptions} options
     */
    async load(options: any) {
        const modelClass = options.modelClass ?? this.modelClass;
        if (!modelClass) {
            throw new Error('OnnxInferenceProvider requires a Transformers.js model class before it can load a model.');
        }
        const { modelClass: _modelClass, ...loadOptions } = options;
        return modelClass._from_pretrained(this.modelId, { ...loadOptions, inferenceProvider: this });
    }

    async getSession(
        fileName: string,
        options: any,
        cache_config = false,
        session_name: string | undefined = undefined,
    ) {
        let custom_config = options.config?.['transformers.js_config'] ?? {};
        const selectedDevice = /** @type {import('../../utils/devices.js').DeviceType} */ selectDevice(
            options.device ?? custom_config.device,
            fileName,
            {
                warn: (msg) => logger.info(msg),
            },
        );
        const executionProviders = deviceToExecutionProviders(selectedDevice);

        const device_config = custom_config.device_config ?? {};
        if (Object.hasOwn(device_config, selectedDevice)) {
            custom_config = { ...custom_config, ...device_config[selectedDevice] };
        }

        const selectedDtype = selectDtype(options.dtype ?? custom_config.dtype, fileName, selectedDevice, {
            configDtype: custom_config.dtype,
            warn: (msg) => logger.info(msg),
        });
        if (!Object.hasOwn(DEFAULT_DTYPE_SUFFIX_MAPPING, selectedDtype)) {
            throw new Error(`Invalid dtype: ${selectedDtype}. Should be one of: ${Object.keys(DATA_TYPES).join(', ')}`);
        }
        if (
            selectedDevice === 'webgpu' &&
            !apis.IS_NODE_ENV &&
            selectedDtype === DATA_TYPES.fp16 &&
            !(await isWebGpuFp16Supported())
        ) {
            throw new Error(`The device (${selectedDevice}) does not support fp16.`);
        }

        const suffix = DEFAULT_DTYPE_SUFFIX_MAPPING[selectedDtype];
        const session_options = { ...options.session_options };
        session_options.executionProviders ??= executionProviders;

        const free_dimension_overrides = custom_config.free_dimension_overrides;
        if (free_dimension_overrides) {
            session_options.freeDimensionOverrides ??= free_dimension_overrides;
        } else if (selectedDevice.startsWith('webnn') && !session_options.freeDimensionOverrides) {
            logger.warn(
                `WebNN does not currently support dynamic shapes and requires 'free_dimension_overrides' to be set in config.json, preferably as a field within config["transformers.js_config"]["device_config"]["${selectedDevice}"]. ` +
                    `When 'free_dimension_overrides' is not set, you may experience significant performance degradation.`,
            );
        }

        const bufferOrPathPromise = getCoreModelFile(this.modelId, fileName, options, suffix);
        const use_external_data_format = options.use_external_data_format ?? custom_config.use_external_data_format;
        const externalData = await getModelDataFiles(
            this.modelId,
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
            const names = getOnnxProviderHost().getCacheNames(options.config, { prefix: 'present', session_name });
            if (names.size > 0 && !isONNXProxy()) {
                const preferredOutputLocation = {};
                for (const key of names) preferredOutputLocation[key] = 'gpu-buffer';
                session_options.preferredOutputLocation = preferredOutputLocation;
            }
        }

        return {
            buffer_or_path: await bufferOrPathPromise,
            session_options,
            session_config: { dtype: selectedDtype, device: selectedDevice },
        };
    }

    async constructSessions(names: Record<string, string>, options: any, cache_sessions: any = undefined) {
        return Object.fromEntries(
            await Promise.all(
                Object.keys(names).map(async (name) => {
                    const sessionInfo = await this.getSession(
                        names[name],
                        options,
                        cache_sessions?.[name] ?? false,
                        name,
                    );
                    const ortSession = await createInferenceSession(
                        sessionInfo.buffer_or_path,
                        sessionInfo.session_options,
                        sessionInfo.session_config,
                    );
                    const session = {
                        inputNames: ortSession.inputNames,
                        outputNames: ortSession.outputNames,
                        inputMetadata: ortSession.inputMetadata,
                        outputMetadata: ortSession.outputMetadata,
                        config: (ortSession as any).config,
                        run: (inputs: any) => this.run(ortSession, inputs),
                        release: () => ortSession.release(),
                    };
                    return [name, session];
                }),
            ),
        );
    }

    async run(session: any, inputs: Record<string, any>) {
        const checkedInputs = validateInputs(session, inputs);
        try {
            const ortFeed = Object.fromEntries(
                Object.entries(checkedInputs).map(([key, value]) => {
                    const storage = getOnnxProviderHost().getBackendTensorStorage(value);
                    const tensor: any =
                        storage?.backend === 'onnx'
                            ? storage.handle
                            : new OrtTensor(value.type, value.data, value.dims);
                    if (
                        apis.IS_NODE_ENV &&
                        typeof Float16Array !== 'undefined' &&
                        tensor.cpuData instanceof Float16Array
                    ) {
                        tensor.cpuData = new Uint16Array(tensor.cpuData.buffer);
                    }
                    return [key, tensor];
                }),
            );
            return replaceTensors(await runInferenceSession(session, ortFeed));
        } catch (error) {
            const formatted = Object.fromEntries(
                Object.entries(checkedInputs).map(([key, tensor]) => {
                    const unpacked: any = { type: tensor.type, dims: tensor.dims, location: tensor.location };
                    if (unpacked.location !== 'gpu-buffer') unpacked.data = tensor.data;
                    return [key, unpacked];
                }),
            );
            logger.error(`An error occurred during model execution: "${error}".`);
            logger.error('Inputs given to model:', formatted);
            throw error;
        }
    }
}

function replaceTensors(value: any): any {
    for (const property in value) {
        if (isONNXTensor(value[property])) {
            const tensor = value[property];
            value[property] = getOnnxProviderHost().createBackendTensor({
                backend: 'onnx',
                handle: tensor,
                get type() {
                    return tensor.type;
                },
                get dims() {
                    return tensor.dims;
                },
                set dims(value) {
                    tensor.dims = value;
                },
                get data() {
                    return tensor.data;
                },
                get size() {
                    return tensor.size;
                },
                get location() {
                    return tensor.location;
                },
                dispose() {
                    tensor.dispose();
                },
            });
        } else if (value[property] && typeof value[property] === 'object') {
            replaceTensors(value[property]);
        }
    }
    return value;
}

function validateInputs(session: any, inputs: Record<string, any>): Record<string, any> {
    const checkedInputs = Object.create(null);
    const missingInputs = [];
    for (const inputName of session.inputNames) {
        const tensor = inputs[inputName];
        if (!tensor || !Array.isArray(tensor.dims) || typeof tensor.type !== 'string') {
            missingInputs.push(inputName);
            continue;
        }
        checkedInputs[inputName] = isONNXProxy() ? tensor.clone() : tensor;
    }
    if (missingInputs.length > 0) {
        throw new Error(
            `An error occurred during model execution: "Missing the following inputs: ${missingInputs.join(', ')}.`,
        );
    }

    const inputNames = Object.keys(inputs);
    if (inputNames.length > session.inputNames.length) {
        const ignored = inputNames.filter((inputName) => !session.inputNames.includes(inputName));
        logger.warn(
            `WARNING: Too many inputs were provided (${inputNames.length} > ${session.inputNames.length}). The following inputs will be ignored: "${ignored.join(', ')}".`,
        );
    }
    return checkedInputs;
}

async function getCoreModelFile(modelId: string, fileName: string, options: any, suffix: string): Promise<any> {
    const baseName = `${fileName}${suffix}.onnx`;
    const subfolder = options.subfolder ?? 'onnx';
    return getOnnxProviderHost().getModelFile(
        modelId,
        subfolder ? `${subfolder}/${baseName}` : baseName,
        true,
        options,
        apis.IS_NODE_ENV,
    );
}

function externalDataChunkNames(fullName: string, count: number): string[] {
    return Array.from({ length: count }, (_, index) => `${fullName}_data${index === 0 ? '' : `_${index}`}`);
}

async function getModelDataFiles(
    modelId: string,
    fileName: string,
    suffix: string,
    options: any,
    externalConfig: any,
    sessionOptions: any,
): Promise<any[]> {
    const baseName = `${fileName}${suffix}.onnx`;
    let count = 0;
    if (typeof externalConfig === 'object' && externalConfig !== null) {
        count = +(externalConfig[baseName] ?? externalConfig[fileName] ?? 0);
    } else if (externalConfig) {
        count = +externalConfig;
    }
    if (count > 1024)
        throw new Error(`The number of external data chunks (${count}) exceeds the maximum allowed value (1024).`);

    if (count > 0) {
        const subfolder = options.subfolder ?? 'onnx';
        return Promise.all(
            externalDataChunkNames(baseName, count).map(async (path) => {
                const data = await getOnnxProviderHost().getModelFile(
                    modelId,
                    subfolder ? `${subfolder}/${path}` : path,
                    true,
                    options,
                    apis.IS_NODE_ENV,
                );
                return data instanceof Uint8Array ? { path, data } : path;
            }),
        );
    }
    if (sessionOptions.externalData !== undefined) {
        return Promise.all(
            sessionOptions.externalData.map(async (item: any) =>
                typeof item.data === 'string'
                    ? { ...item, data: await getOnnxProviderHost().getModelFile(modelId, item.data, true, options) }
                    : item,
            ),
        );
    }
    return [];
}
