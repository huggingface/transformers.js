import type { DataType, DeviceType } from '@huggingface/transformers';

export interface ModelConfig {
    modelId: string;
    revision?: string;
    device?: DeviceType;
    dtype?: DataType | Record<string, DataType>;
}
