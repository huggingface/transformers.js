import { OnnxInferenceProvider, OnnxTensorOpRegistry, configureOnnxProviderHost } from '@huggingface/transformers-onnx';

import { env, apis } from '../env.js';
import { logger } from '../utils/logger.js';
import { getModelFile } from '../utils/hub.js';
import { getCacheNames } from '../configs.js';
import { Tensor } from '../utils/tensor.js';
import { TensorOpRegistry } from '../ops/registry.js';
import { getCache } from '../utils/cache.js';

configureOnnxProviderHost({
    env,
    apis,
    logger,
    getModelFile,
    getCacheNames,
    createBackendTensor: (storage) => Tensor.fromBackendStorage(storage),
    getBackendTensorStorage: (tensor) => tensor?.getBackendStorage?.() ?? null,
    getCache,
});

TensorOpRegistry.register(OnnxTensorOpRegistry);

export { OnnxInferenceProvider };
