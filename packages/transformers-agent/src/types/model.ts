import type { DataType, DeviceType } from '@huggingface/transformers';

export interface ModelConfig {
    modelId: string;
    device?: DeviceType;
    dtype?: DataType | Record<string, DataType>;
}
