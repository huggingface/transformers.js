import type { DataType, DeviceType, ProgressCallback } from '@huggingface/transformers';
import type { AgentConfig } from './interfaces.d.ts';
import type { Agent } from './Agent.ts';

export interface ModelConfig {
    modelId: string;
    device?: DeviceType;
    /** Quantization dtype, or a per-module dtype map for encoder-decoder models. */
    dtype?: DataType | Record<string, DataType>;
}

export class Model {
    readonly modelId: string;
    readonly device: DeviceType;
    readonly dtype: DataType | Record<string, DataType>;
    readonly isInitialized: boolean = false;

    constructor(config: ModelConfig) {
        this.modelId = config.modelId;
        this.device = config.device ?? 'cpu';
        this.dtype = config.dtype ?? 'fp32';
    }

    async isCached(): Promise<boolean> {
        throw new Error('Not implemented');
    }

    async downloadSize(): Promise<number> {
        throw new Error('Not implemented');
    }

    async cachedSize(): Promise<number> {
        throw new Error('Not implemented');
    }

    async init(progressCallback?: ProgressCallback): Promise<void> {
        throw new Error('Not implemented');
    }

    static async load(config: ModelConfig, progressCallback?: ProgressCallback): Promise<Model> {
        throw new Error('Not implemented');
    }

    agent(options: Omit<AgentConfig, 'model'>): Agent {
        throw new Error('Not implemented');
    }
}
