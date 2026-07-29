import { env, apis } from '../env.js';
import { logger } from '../utils/logger.js';
import { getModelFile, MAX_EXTERNAL_DATA_CHUNKS } from '../utils/hub.js';
import { getCacheNames } from '../configs.js';
import { Tensor } from '../utils/tensor.js';
import { TensorOpRegistry } from '../ops/registry.js';
import { getCache } from '../utils/cache.js';

const ONNX_HOST_SYMBOL = Symbol.for('transformers.js.onnxProviderHost');
const host = {
    env,
    apis,
    logger,
    getModelFile,
    getCacheNames,
    createBackendTensor: (storage) => Tensor.fromBackendStorage(storage),
    getBackendTensorStorage: (tensor) => tensor?.getBackendStorage?.() ?? null,
    getCache,
    maxExternalDataChunks: MAX_EXTERNAL_DATA_CHUNKS,
};

let modulePromise;

export function getOnnxProviderModule() {
    if (!modulePromise) {
        globalThis[ONNX_HOST_SYMBOL] = host;
        const pending = import('@huggingface/transformers-onnx')
            .then((module) => {
                module.configureOnnxProviderHost(host);
                TensorOpRegistry.register(module.OnnxTensorOpRegistry);
                return module;
            })
            .catch((error) => {
                if (modulePromise === pending) modulePromise = undefined;
                throw error;
            });
        modulePromise = pending;
    }
    return modulePromise;
}

export async function getDefaultInferenceProvider(modelId) {
    const { OnnxInferenceProvider } = await getOnnxProviderModule();
    return OnnxInferenceProvider.from_modelId(modelId);
}
