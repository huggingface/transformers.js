import { getGlobalLiteRtPromise, loadAndCompile, loadLiteRt, type Accelerator } from '@litertjs/core';
import {
    LITE_RT_MODEL_DESCRIPTORS,
    resolveModelDescriptor,
    selectArtifact,
    type LiteRtArtifact,
    type LiteRtModelDescriptor,
} from './descriptors.js';
import { createVisionModel, validateCompiledModel } from './vision.js';
import { createWhisperModel, type GenerationConfig } from './whisper.js';
import { createYoloModel } from './yolo.js';

const DEFAULT_WASM_PATH = 'https://cdn.jsdelivr.net/npm/@litertjs/core@2.5.3/wasm/';

export type LiteRtInferenceProviderOptions = {
    artifactModelId?: string;
    sharedModelId?: string;
    modelFile?: string;
    revision?: string;
    sharedRevision?: string;
    wasmPath?: string;
    jspi?: boolean;
};

function selectAccelerator(device: unknown): Accelerator | 'webnn' {
    if (device == null || device === 'cpu' || device === 'wasm') return 'wasm';
    if (device === 'webgpu' || device === 'webnn') return device;
    throw new Error(`LiteRT.js does not support device "${String(device)}".`);
}

function getSupportedDtypes(descriptor: LiteRtModelDescriptor): string[] {
    const dtypes = Object.keys(descriptor.artifacts);
    if (dtypes.includes('q8')) dtypes.push('int8');
    return dtypes;
}

function getArtifactOptions(options: Record<string, any>, revision: string): Record<string, any> {
    const { inferenceBackend: _inferenceBackend, inferenceProvider: _inferenceProvider, ...artifactOptions } = options;
    return { ...artifactOptions, revision, subfolder: null };
}

/**
 * Experimental LiteRT.js provider for selected fixed-shape models.
 */
export class LiteRtInferenceProvider {
    readonly modelId: string;
    readonly artifactModelId: string;
    readonly capabilities: { devices: readonly string[]; dtypes: readonly string[]; tasks: readonly string[] };
    readonly sharedAssets?: { revision?: string };
    readonly descriptor: LiteRtModelDescriptor;
    readonly revision?: string;
    readonly modelFile?: string;
    readonly wasmPath: string;
    readonly jspi: boolean;

    static from_modelId(modelId: string, options: LiteRtInferenceProviderOptions = {}): LiteRtInferenceProvider {
        return new LiteRtInferenceProvider(modelId, options);
    }

    constructor(modelId: string, options: LiteRtInferenceProviderOptions = {}) {
        this.descriptor = resolveModelDescriptor(modelId);
        this.artifactModelId = options.artifactModelId ?? this.descriptor.artifactModelId;
        this.modelId = options.sharedModelId ?? this.descriptor.sharedModelId;
        this.revision = options.revision;
        this.modelFile = options.modelFile;
        this.wasmPath = options.wasmPath ?? DEFAULT_WASM_PATH;
        this.jspi = options.jspi ?? false;
        this.capabilities = {
            devices: ['wasm', 'webgpu', 'webnn'],
            dtypes: getSupportedDtypes(this.descriptor),
            tasks: [this.descriptor.task],
        };
        const sharedRevision = options.sharedRevision ?? this.descriptor.sharedRevision;
        if (sharedRevision) this.sharedAssets = { revision: sharedRevision };
    }

    private getArtifact(options: Record<string, unknown>): LiteRtArtifact {
        const artifact = selectArtifact(this.descriptor, options.dtype, this.modelFile);
        return { ...artifact, revision: this.revision ?? artifact.revision };
    }

    listModelArtifacts(options: Record<string, unknown>): string[] {
        const files = ['config.json', this.getArtifact(options).file];
        if (this.descriptor.execution === 'whisper') files.push('generation_config.json');
        return files;
    }

    async getModelArtifactMetadata(
        file: string,
        options: Record<string, any>,
    ): Promise<{ size?: number; fromCache?: boolean } | null> {
        const artifact = this.getArtifact(options);
        if (file !== artifact.file) return null;
        if (!options.getModelFileMetadata) {
            throw new Error('Transformers.js did not provide `getModelFileMetadata()`.');
        }
        const metadata = await options.getModelFileMetadata(
            this.artifactModelId,
            file,
            getArtifactOptions(options, artifact.revision),
        );
        return metadata.exists ? { size: metadata.size, fromCache: metadata.fromCache } : null;
    }

    async deleteModelArtifact(file: string, options: Record<string, any>): Promise<boolean> {
        const artifact = this.getArtifact(options);
        if (file !== artifact.file) return false;
        if (!options.deleteModelFile) throw new Error('Transformers.js did not provide `deleteModelFile()`.');
        return options.deleteModelFile(this.artifactModelId, file, getArtifactOptions(options, artifact.revision));
    }

    async load(options: Record<string, any>): Promise<Function> {
        if (options.task && options.task !== this.descriptor.task) {
            throw new Error(
                `LiteRtInferenceProvider model "${this.descriptor.id}" does not support the "${options.task}" task.`,
            );
        }
        const pendingRuntime = getGlobalLiteRtPromise();
        if (pendingRuntime) await pendingRuntime;
        else await loadLiteRt(this.wasmPath, { jspi: this.jspi });

        const artifact = this.getArtifact(options);
        if (!options.getModelFile) throw new Error('Transformers.js did not provide `getModelFile()`.');
        const modelBytes = await options.getModelFile(
            this.artifactModelId,
            artifact.file,
            true,
            getArtifactOptions(options, artifact.revision),
        );
        if (!(modelBytes instanceof Uint8Array)) {
            throw new TypeError('Transformers.js `getModelFile()` must return model bytes in browser environments.');
        }
        const compiled = await loadAndCompile(modelBytes, {
            accelerator: selectAccelerator(options.device),
            ...(options.session_options ?? {}),
        });
        try {
            if (this.descriptor.execution !== 'whisper') validateCompiledModel(compiled, this.descriptor);
            if (this.descriptor.execution === 'whisper') {
                return createWhisperModel(compiled, (options.generation_config ?? {}) as GenerationConfig);
            }
            if (this.descriptor.execution === 'yolo-detection') return createYoloModel(compiled, this.descriptor);
            return createVisionModel(compiled, this.descriptor);
        } catch (error) {
            compiled.delete();
            throw error;
        }
    }
}

export { LITE_RT_MODEL_DESCRIPTORS, resolveModelDescriptor, selectArtifact } from './descriptors.js';
export { postProcessYolo, preprocessYolo } from './yolo.js';
export { visionTesting } from './vision.js';
