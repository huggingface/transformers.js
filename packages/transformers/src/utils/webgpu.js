import { env } from '../env.js';
import { getTensorConstructor } from '../backends/onnx.js';

/**
 * Select aligned tensor rows without transferring WebGPU data to the CPU.
 * Returns null when the tensor cannot use WebGPU buffer copies.
 * @param {import('onnxruntime-common').Tensor} tensor
 * @param {number[]} indices
 * @returns {Promise<import('onnxruntime-common').Tensor|null>}
 * @internal
 */
export async function selectWebGPURows(tensor, indices) {
    if (typeof GPUBufferUsage === 'undefined') return null;
    const TypedArray = {
        float32: Float32Array,
        float16: Uint16Array,
        int32: Int32Array,
        uint32: Uint32Array,
        int64: BigInt64Array,
        uint8: Uint8Array,
        bool: Uint8Array,
    }[tensor.type];
    const row_size = tensor.dims.slice(1).reduce((a, b) => a * b, 1);
    const row_bytes = row_size * TypedArray?.BYTES_PER_ELEMENT;
    // Buffer copies require four-byte aligned offsets and lengths. Odd fp16
    // rows and packed data types keep the existing CPU fallback.
    if (!row_bytes || row_bytes % 4 !== 0 || indices.length === 0) return null;
    if (indices.some((index) => !Number.isInteger(index) || index < 0 || index >= tensor.dims[0])) return null;
    const source = tensor.gpuBuffer;
    if (!(source.usage & GPUBufferUsage.COPY_SRC)) return null;
    const device = await env.backends.onnx.webgpu?.device;
    if (!device) return null;
    const byte_length = indices.length * row_bytes;
    // ONNX Runtime rounds GPU transfer sizes up to sixteen bytes.
    const size = Math.ceil(byte_length / 16) * 16;
    if (size > device.limits.maxBufferSize) return null;
    const output = device.createBuffer({
        label: 'transformers-beam-rows',
        size,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    try {
        const encoder = device.createCommandEncoder();
        for (let row = 0; row < indices.length; ++row) {
            encoder.copyBufferToBuffer(source, indices[row] * row_bytes, output, row * row_bytes, row_bytes);
        }
        device.queue.submit([encoder.finish()]);
        return getTensorConstructor().fromGpuBuffer(output, {
            dataType: /** @type {import('onnxruntime-common').Tensor.GpuBufferDataTypes} */ (tensor.type),
            dims: [indices.length, ...tensor.dims.slice(1)],
            dispose: () => output.destroy(),
            download: async () => {
                const staging = device.createBuffer({
                    label: 'transformers-beam-readback',
                    size,
                    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
                });
                try {
                    const copy = device.createCommandEncoder();
                    copy.copyBufferToBuffer(output, 0, staging, 0, size);
                    device.queue.submit([copy.finish()]);
                    await staging.mapAsync(GPUMapMode.READ);
                    return new TypedArray(staging.getMappedRange().slice(0, byte_length));
                } finally {
                    staging.destroy();
                }
            },
        });
    } catch (error) {
        output.destroy();
        throw error;
    }
}
