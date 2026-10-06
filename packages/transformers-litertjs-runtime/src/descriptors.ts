export type LiteRtArtifact = {
    file: string;
    revision: string;
    sha256?: string;
};

export type TensorContract = {
    name: string;
    dtype: 'float32' | 'int32';
    shape: readonly number[];
};

export type LiteRtModelDescriptor = {
    id: string;
    artifactModelId: string;
    sharedModelId: string;
    task: 'image-classification' | 'automatic-speech-recognition' | 'object-detection';
    execution: 'vision-classification' | 'whisper' | 'yolo-detection';
    artifacts: Readonly<Partial<Record<'fp32' | 'q8', LiteRtArtifact>>>;
    defaultDtype: 'fp32' | 'q8';
    inputs: readonly TensorContract[];
    outputs: readonly TensorContract[];
    inputLayout?: 'nchw' | 'nhwc';
    sourceInputShape?: readonly [number, number, number, number];
    sharedRevision?: string;
};

export const LITE_RT_MODEL_DESCRIPTORS: Readonly<Record<string, LiteRtModelDescriptor>> = Object.freeze({
    'litert-community/whisper-tiny': {
        id: 'whisper-tiny',
        artifactModelId: 'litert-community/whisper-tiny',
        sharedModelId: 'openai/whisper-tiny',
        task: 'automatic-speech-recognition',
        execution: 'whisper',
        artifacts: {
            fp32: { file: 'whisper_tiny_30s_f32.tflite', revision: 'main' },
            q8: { file: 'whisper_tiny_30s_i8.tflite', revision: 'main' },
        },
        defaultDtype: 'q8',
        inputs: [{ name: 'input_features', dtype: 'float32', shape: [1, 80, 3000] }],
        outputs: [{ name: 'tokens', dtype: 'int32', shape: [1, 128] }],
    },
    'litert-community/whisper-base': {
        id: 'whisper-base',
        artifactModelId: 'litert-community/whisper-base',
        sharedModelId: 'openai/whisper-base',
        task: 'automatic-speech-recognition',
        execution: 'whisper',
        artifacts: {
            fp32: {
                file: 'whisper_base_30s_f32.tflite',
                revision: 'ba2c613646edc2b72f5a51fa0b5b0b322e436101',
                sha256: '83f90febae74f6db2b671d792ce844f93de5dfa6d6743cdba8c7a208c8dd708a',
            },
            q8: {
                file: 'whisper_base_30s_i8.tflite',
                revision: 'ba2c613646edc2b72f5a51fa0b5b0b322e436101',
                sha256: 'f6943d9d293138850b729e074057956c664891c57837692b1bac4608c4506cd1',
            },
        },
        defaultDtype: 'q8',
        inputs: [{ name: 'input_features', dtype: 'float32', shape: [1, 80, 3000] }],
        outputs: [{ name: 'tokens', dtype: 'int32', shape: [1, 128] }],
    },
    'litert-community/efficientnet_b0': {
        id: 'efficientnet-b0',
        artifactModelId: 'litert-community/efficientnet_b0',
        sharedModelId: 'microsoft/resnet-50',
        task: 'image-classification',
        execution: 'vision-classification',
        artifacts: {
            fp32: {
                file: 'efficientnet_b0.tflite',
                revision: '2f882b5d920c156a466bcb5debae63af78d5fc08',
                sha256: '9b58d40a42fc7996070de6256a3a772771885a2609fc97753d054a3e6ea1ea98',
            },
            q8: {
                file: 'efficientnet_b0_weight_only_wi8_afp32.tflite',
                revision: '2f882b5d920c156a466bcb5debae63af78d5fc08',
            },
        },
        defaultDtype: 'fp32',
        inputs: [{ name: 'pixel_values', dtype: 'float32', shape: [1, 3, 224, 224] }],
        outputs: [{ name: 'logits', dtype: 'float32', shape: [1, 1000] }],
        inputLayout: 'nchw',
        sourceInputShape: [1, 3, 224, 224],
    },
    'litert-community/inception_v3': {
        id: 'inception-v3',
        artifactModelId: 'litert-community/inception_v3',
        sharedModelId: 'microsoft/resnet-50',
        task: 'image-classification',
        execution: 'vision-classification',
        artifacts: {
            fp32: {
                file: 'inception_v3.tflite',
                revision: '8adf46014deda51d76ea19693140660c7610b65e',
                sha256: '0116bb2d6fa277da4de65b33c11a8fa870840531dbcc7bda0aeccbecd3299dc1',
            },
        },
        defaultDtype: 'fp32',
        inputs: [{ name: 'pixel_values', dtype: 'float32', shape: [1, 299, 299, 3] }],
        outputs: [{ name: 'logits', dtype: 'float32', shape: [1, 1000] }],
        inputLayout: 'nhwc',
        sourceInputShape: [1, 3, 299, 299],
    },
    'litert-community/resnet50': {
        id: 'resnet-50',
        artifactModelId: 'litert-community/resnet50',
        sharedModelId: 'microsoft/resnet-50',
        task: 'image-classification',
        execution: 'vision-classification',
        artifacts: {
            fp32: {
                file: 'resnet50.tflite',
                revision: 'fdc3dccce8ea6021febcc7c4027c95134d9bb9d5',
                sha256: 'b60ece2473934776b8de38244d1fc5d107ecebabd7080a1a19ea8c28a4643be5',
            },
            q8: {
                file: 'resnet50_dynamic_wi8_afp32.tflite',
                revision: 'fdc3dccce8ea6021febcc7c4027c95134d9bb9d5',
            },
        },
        defaultDtype: 'fp32',
        inputs: [{ name: 'pixel_values', dtype: 'float32', shape: [1, 3, 224, 224] }],
        outputs: [{ name: 'logits', dtype: 'float32', shape: [1, 1000] }],
        inputLayout: 'nchw',
        sourceInputShape: [1, 3, 224, 224],
    },
    'litert-community/yolo26n': {
        id: 'yolo26n-w8a32-v0.6.6',
        artifactModelId: 'https://github.com/ultralytics/yolo-flutter-app/releases/download/v0.6.6',
        sharedModelId: 'onnx-community/yolo26n-ONNX',
        task: 'object-detection',
        execution: 'yolo-detection',
        artifacts: {
            q8: {
                file: 'yolo26n_w8a32.tflite',
                revision: 'v0.6.6',
                sha256: 'd9cef07ce652ccfa9ce58e4ac8a4df98ff037739a9dad20a8afcae21b545df73',
            },
        },
        defaultDtype: 'q8',
        inputs: [{ name: 'pixel_values', dtype: 'float32', shape: [1, 3, 640, 640] }],
        outputs: [{ name: 'predictions', dtype: 'float32', shape: [1, 84, 8400] }],
        inputLayout: 'nchw',
        sourceInputShape: [1, 3, 640, 640],
    },
});

const ALIASES: Readonly<Record<string, string>> = Object.freeze({
    efficientnet: 'litert-community/efficientnet_b0',
    efficientnet_b0: 'litert-community/efficientnet_b0',
    inception_v3: 'litert-community/inception_v3',
    resnet50: 'litert-community/resnet50',
    whisper_base: 'litert-community/whisper-base',
    whisper_tiny: 'litert-community/whisper-tiny',
    yolo26n: 'litert-community/yolo26n',
});

export function resolveModelDescriptor(modelId: string): LiteRtModelDescriptor {
    const key = ALIASES[modelId] ?? ALIASES[modelId.split('/').at(-1) ?? ''] ?? modelId;
    const descriptor = LITE_RT_MODEL_DESCRIPTORS[key];
    if (!descriptor) {
        throw new Error(
            `Unsupported LiteRT.js model "${modelId}". Supported models: ${Object.keys(LITE_RT_MODEL_DESCRIPTORS).join(', ')}.`,
        );
    }
    return descriptor;
}

export function selectArtifact(descriptor: LiteRtModelDescriptor, dtype: unknown, override?: string): LiteRtArtifact {
    const selectedDtype =
        dtype === 'int8' || dtype === 'q8' ? 'q8' : dtype === 'auto' || dtype == null ? descriptor.defaultDtype : dtype;
    const artifact = descriptor.artifacts[selectedDtype as 'fp32' | 'q8'];
    if (!artifact) {
        throw new Error(
            `LiteRT.js model "${descriptor.id}" does not provide dtype "${String(dtype)}". Available dtypes: ${Object.keys(descriptor.artifacts).join(', ')}.`,
        );
    }
    return override ? { ...artifact, file: override } : artifact;
}
