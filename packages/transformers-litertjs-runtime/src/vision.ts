import { Tensor } from '@huggingface/transformers';
import { Tensor as LiteRtTensor, type CompiledModel } from '@litertjs/core';
import type { LiteRtModelDescriptor, TensorContract } from './descriptors.js';

function sameShape(actual: readonly number[], expected: readonly number[]): boolean {
    return actual.length === expected.length && actual.every((value, index) => value === expected[index]);
}

export function validateCompiledModel(model: CompiledModel, descriptor: LiteRtModelDescriptor): void {
    const inputs = model.getInputDetails();
    const outputs = model.getOutputDetails();
    validateDetails('input', descriptor, inputs, descriptor.inputs);
    validateDetails('output', descriptor, outputs, descriptor.outputs);
}

function validateDetails(
    kind: 'input' | 'output',
    descriptor: LiteRtModelDescriptor,
    actual: readonly { name: string; dtype: string; shape: Int32Array }[],
    expected: readonly TensorContract[],
): void {
    if (actual.length !== expected.length) {
        throw new Error(
            `LiteRT.js model "${descriptor.id}" expected ${expected.length} ${kind} tensor(s), received ${actual.length}.`,
        );
    }
    actual.forEach((detail, index) => {
        const contract = expected[index];
        if (detail.dtype !== contract.dtype || !sameShape(Array.from(detail.shape), contract.shape)) {
            throw new Error(
                `LiteRT.js model "${descriptor.id}" ${kind} "${contract.name}" expected ${contract.dtype} [${contract.shape}], received ${detail.dtype} [${Array.from(detail.shape)}] from "${detail.name}".`,
            );
        }
    });
}

function resizeNchw(
    data: Float32Array,
    dims: readonly number[],
    targetHeight: number,
    targetWidth: number,
): Float32Array {
    const [batch, channels, sourceHeight, sourceWidth] = dims;
    if (batch !== 1) throw new Error('LiteRT.js fixed-shape vision models currently support batch size 1.');
    if (sourceHeight === targetHeight && sourceWidth === targetWidth) return data;
    const output = new Float32Array(channels * targetHeight * targetWidth);
    for (let channel = 0; channel < channels; ++channel) {
        const sourceOffset = channel * sourceHeight * sourceWidth;
        const targetOffset = channel * targetHeight * targetWidth;
        for (let y = 0; y < targetHeight; ++y) {
            const sourceY = Math.min(sourceHeight - 1, Math.round(((y + 0.5) * sourceHeight) / targetHeight - 0.5));
            for (let x = 0; x < targetWidth; ++x) {
                const sourceX = Math.min(sourceWidth - 1, Math.round(((x + 0.5) * sourceWidth) / targetWidth - 0.5));
                output[targetOffset + y * targetWidth + x] = data[sourceOffset + sourceY * sourceWidth + sourceX];
            }
        }
    }
    return output;
}

function nchwToNhwc(data: Float32Array, dims: readonly number[]): Float32Array {
    const [batch, channels, height, width] = dims;
    if (batch !== 1) throw new Error('LiteRT.js fixed-shape vision models currently support batch size 1.');
    const output = new Float32Array(data.length);
    for (let y = 0; y < height; ++y) {
        for (let x = 0; x < width; ++x) {
            for (let channel = 0; channel < channels; ++channel) {
                output[(y * width + x) * channels + channel] = data[(channel * height + y) * width + x];
            }
        }
    }
    return output;
}

export async function runVisionModel(
    model: CompiledModel,
    descriptor: LiteRtModelDescriptor,
    inputs: Record<string, Tensor>,
): Promise<Record<string, Tensor>> {
    const input = inputs.pixel_values;
    if (!input || input.type !== 'float32' || input.dims.length !== 4) {
        throw new TypeError(`LiteRT.js model "${descriptor.id}" requires a float32 \`pixel_values\` tensor.`);
    }
    const source = input.data as Float32Array;
    const targetShape = descriptor.sourceInputShape ?? (descriptor.inputs[0].shape as [number, number, number, number]);
    let data = resizeNchw(source, input.dims, targetShape[2], targetShape[3]);
    let dims = Array.from(targetShape);
    if (descriptor.inputLayout === 'nhwc') {
        data = nchwToNhwc(data, dims);
        dims = [dims[0], dims[2], dims[3], dims[1]];
    }

    const runtimeInput = new LiteRtTensor(data, dims);
    let runtimeOutputs: LiteRtTensor[] | undefined;
    try {
        runtimeOutputs = (await model.run([runtimeInput])) as LiteRtTensor[];
        return Object.fromEntries(
            await Promise.all(
                descriptor.outputs.map(async (contract, index) => {
                    const output = runtimeOutputs?.[index];
                    if (!output)
                        throw new Error(`LiteRT.js model "${descriptor.id}" did not return "${contract.name}".`);
                    const outputData = await output.data();
                    return [contract.name, new Tensor(contract.dtype, outputData.slice(), Array.from(contract.shape))];
                }),
            ),
        );
    } finally {
        runtimeInput.delete();
        if (runtimeOutputs) for (const output of runtimeOutputs) output.delete();
    }
}

export function createVisionModel(model: CompiledModel, descriptor: LiteRtModelDescriptor): Function {
    const forward = (inputs: Record<string, Tensor>) => runVisionModel(model, descriptor, inputs);
    return Object.assign(forward, {
        forward,
        capabilities: { forward: { version: 1 } },
        dispose: () => model.delete(),
    });
}

export const visionTesting = { nchwToNhwc, resizeNchw };
