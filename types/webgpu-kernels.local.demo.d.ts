import type { WebGPUKernelsForwardModel, WebGPUKernelsTensorMap, WebGPUKernelsTextGenerationModel } from './webgpu-kernels.local';
type PipelineForwardModel = {
    (inputs: WebGPUKernelsTensorMap): Promise<WebGPUKernelsTensorMap>;
    readonly config?: Record<string, unknown>;
    forward(inputs: WebGPUKernelsTensorMap): Promise<WebGPUKernelsTensorMap>;
    dispose(): void | Promise<void>;
};
type PipelineTextGenerationModel = PipelineForwardModel & {
    generate: WebGPUKernelsTextGenerationModel['generate'];
};
export declare function adaptWebGPUKernelsModel(model: WebGPUKernelsTextGenerationModel): PipelineTextGenerationModel;
export declare function adaptWebGPUKernelsModel(model: WebGPUKernelsForwardModel): PipelineForwardModel;
export {};
//# sourceMappingURL=webgpu-kernels.local.demo.d.ts.map