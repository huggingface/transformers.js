import { Tensor } from 'onnxruntime-common';
import { getStaticCacheInfo, setStaticCacheInfo } from '../../utils/static-cache.js';

const TYPES = {
    float32: Float32Array,
    float16: Uint16Array,
    int32: Int32Array,
    int64: BigInt64Array,
    uint32: Uint32Array,
    uint8: Uint8Array,
    bool: Uint8Array,
};
const alignedSize = (size) => Math.ceil(size / 16) * 16;

/** Static KV buffers and decode-only capture; prefill uses gpu_graph_id=-1. @private */
export class DecodeGraphCaptureSession {
    /**
     * @param {import('onnxruntime-common').InferenceSession} session ORT session with capture configured in the C++ EP.
     * @param {GPUDevice} device ORT's own device.
     * @param {Object} options
     * @param {string[]} options.cacheNames
     * @param {number} options.maxCacheLength
     * @param {boolean} [options.enableGraphCapture=false]
     * @param {typeof Tensor} [options.TensorConstructor]
     */
    constructor(
        session,
        device,
        { cacheNames, maxCacheLength, enableGraphCapture = false, TensorConstructor = Tensor },
    ) {
        if (!Number.isSafeInteger(maxCacheLength) || maxCacheLength < 1)
            throw new Error('max_cache_length must be a positive integer.');
        if (
            !session.inputNames.includes('input_ids') ||
            !session.inputNames.includes('attention_mask') ||
            !cacheNames.length
        ) {
            throw new Error(
                'Decode graph capture requires input_ids, attention_mask and a static-cache-compatible decoder export.',
            );
        }
        this.session = session;
        this.device = device;
        this.Tensor = TensorConstructor;
        this.maxCacheLength = maxCacheLength;
        this.enableGraphCapture = enableGraphCapture;
        this.cacheNames = cacheNames;
        this.epoch = 0;
        this.disposed = false;
        /** @type {Map<string, Tensor>} */
        this.cache = new Map();
        /** @type {Map<string, Tensor>} */
        this.decodeInputs = new Map();
        /** @type {Map<string, Tensor>} */
        this.decodeOutputs = new Map();
    }

    /** @param {Record<string, Tensor>} feeds */
    async run(feeds) {
        if (this.disposed) throw new Error('Cannot run a disposed decode graph capture session.');
        const ids = feeds.input_ids;
        const mask = feeds.attention_mask;
        if (ids?.dims.length !== 2 || ids.dims[0] !== 1 || ids.dims[1] < 1) {
            throw new Error('Decode graph capture currently requires batch size 1 and non-empty input_ids.');
        }
        if (mask?.dims.length !== 2 || mask.dims[0] !== 1 || !['cpu', 'cpu-pinned'].includes(mask.location)) {
            throw new Error('Decode graph capture requires a CPU attention_mask with batch size 1.');
        }
        const past = feeds[this.cacheNames[0]];
        const info = past && getStaticCacheInfo(past);
        const pastLength = info?.length ?? past?.dims.at(-2);
        if (!Number.isSafeInteger(pastLength) || pastLength < 0)
            throw new Error('Missing or invalid past key values for decode graph capture.');
        const length = pastLength + ids.dims[1];
        if (length > this.maxCacheLength) {
            throw new Error(
                `Static KV cache capacity (${this.maxCacheLength}) exceeded by ${length} tokens. Increase transformers.js_config.max_cache_length.`,
            );
        }
        if (mask.dims[1] !== length)
            throw new Error(`attention_mask length must equal past length plus input length (${length}).`);
        for (const name of this.cacheNames) {
            const tensor = feeds[name];
            const entry = tensor && getStaticCacheInfo(tensor);
            const meta = this.session.inputMetadata.find((item) => item.name === name);
            if (
                !tensor ||
                !meta?.isTensor ||
                tensor.type !== meta.type ||
                tensor.dims.length !== 4 ||
                tensor.dims[0] !== 1 ||
                tensor.dims[1] < 1 ||
                tensor.dims[3] < 1 ||
                (typeof meta.shape[1] === 'number' && meta.shape[1] !== tensor.dims[1]) ||
                (typeof meta.shape[3] === 'number' && meta.shape[3] !== tensor.dims[3]) ||
                (entry?.length ?? tensor.dims[2]) !== pastLength
            ) {
                throw new Error(`Invalid static KV cache input "${name}".`);
            }
            if (entry?.owner !== info?.owner || entry?.epoch !== info?.epoch)
                throw new Error('Cannot combine KV tensors from different static caches.');
        }
        if (!this.cache.size) {
            try {
                for (const name of this.cacheNames) {
                    const source = feeds[name];
                    const dims = [...source.dims];
                    dims[2] = this.maxCacheLength;
                    this.cache.set(name, this.createTensor(source.type, dims));
                }
            } catch (error) {
                this.clearTensors(this.cache);
                throw error;
            }
        }
        if (pastLength === 0 || info?.owner !== this) {
            // A fresh request invalidates views of the previous mutable static cache.
            ++this.epoch;
            if (pastLength) {
                for (const name of this.cacheNames) this.importCache(feeds[name], this.cache.get(name), pastLength);
            }
        }

        // One-token prompts have pastLength=0. Multi-token cache continuations
        // also use the ordinary prefill session, even after decode was captured.
        const decode = pastLength > 0 && ids.dims[1] === 1;
        let outputs;
        if (decode) {
            await this.prepareDecode(feeds);
            for (const [name, target] of this.decodeInputs) {
                const source = feeds[name];
                if (name === 'attention_mask') {
                    const data = new TYPES[source.type](this.maxCacheLength);
                    data.set(source.data);
                    this.upload(new this.Tensor(source.type, data, target.dims), target);
                } else {
                    this.upload(source, target);
                }
            }
            const decodeFeeds = { ...Object.fromEntries(this.cache), ...Object.fromEntries(this.decodeInputs) };
            outputs = await this.session.run(decodeFeeds, this.fetches(this.decodeOutputs), {
                extra: { gpu_graph_id: this.enableGraphCapture ? '0' : '-1' },
            });
        } else {
            try {
                outputs = await this.session.run({ ...feeds, ...Object.fromEntries(this.cache) }, this.fetches(), {
                    extra: { gpu_graph_id: '-1' },
                });
            } catch (error) {
                throw new Error(
                    'Uncaptured prefill with static KV buffers failed. Graph capture requires a compatible GQA/shared-KV export; ordinary concatenating KV exports are not supported.',
                    { cause: error },
                );
            }
        }
        /** @type {Record<string, Tensor>} */
        const result = {};
        try {
            for (const [name, tensor] of Object.entries(outputs)) {
                const cache = this.cache.get(name.replace(/^present\./, 'past_key_values.'));
                if (cache) {
                    const view = this.view(cache);
                    setStaticCacheInfo(view, this, length);
                    result[name] = view;
                } else if (this.decodeOutputs.get(name) === tensor) {
                    // getData() caches data and changes location to CPU. Download
                    // through a fresh view so the bound tensor stays on the GPU.
                    const view = this.view(tensor);
                    result[name] = new this.Tensor(tensor.type, await view.getData(), tensor.dims);
                    view.dispose();
                } else {
                    result[name] = tensor;
                }
            }
            return result;
        } catch (error) {
            for (const tensor of Object.values(result)) tensor.dispose();
            throw error;
        }
    }

    /** @param {Map<string, Tensor>} [outputs] */
    fetches(outputs = new Map()) {
        return Object.fromEntries(
            this.session.outputNames.map((name) => [
                name,
                this.cache.get(name.replace(/^present\./, 'past_key_values.')) ?? outputs.get(name) ?? null,
            ]),
        );
    }

    /** @param {Record<string, Tensor>} feeds */
    async prepareDecode(feeds) {
        if (this.decodeInputs.size) return;
        /** @type {Record<string, number>} */
        const overrides = {};
        try {
            for (const meta of this.session.inputMetadata) {
                if (!meta.isTensor) throw new Error(`Unsupported non-tensor input: ${meta.name}`);
                const source = this.cache.get(meta.name) ?? feeds[meta.name];
                if (!source) throw new Error(`Missing decode input: ${meta.name}`);
                const dims = [...source.dims];
                if (meta.name === 'attention_mask') dims[1] = this.maxCacheLength;
                for (let i = 0; i < meta.shape.length; ++i) {
                    const symbol = meta.shape[i];
                    if (typeof symbol === 'string') {
                        if (overrides[symbol] !== undefined && overrides[symbol] !== dims[i])
                            throw new Error(`Conflicting decode dimension: ${symbol}`);
                        overrides[symbol] = dims[i];
                    }
                }
                if (!this.cache.has(meta.name)) this.decodeInputs.set(meta.name, this.createTensor(source.type, dims));
            }
            for (const meta of this.session.outputMetadata) {
                if (this.cache.has(meta.name.replace(/^present\./, 'past_key_values.'))) continue;
                if (!meta.isTensor) throw new Error(`Unsupported non-tensor output: ${meta.name}`);
                const dims = meta.shape.map((dim) => (typeof dim === 'number' ? dim : overrides[dim]));
                if (dims.some((dim) => !Number.isSafeInteger(dim) || dim < 1))
                    throw new Error(`Decode output "${meta.name}" must have a fixed shape.`);
                this.decodeOutputs.set(meta.name, this.createTensor(meta.type, dims));
            }
        } catch (error) {
            this.clearTensors(this.decodeInputs);
            this.clearTensors(this.decodeOutputs);
            throw error;
        }
    }

    /** @param {Tensor['type']} type @param {readonly number[]} dims */
    createTensor(type, dims) {
        if (!TYPES[type]) throw new Error(`Unsupported WebGPU tensor type: ${type}`);
        const size = dims.reduce((a, b) => a * b, 1) * TYPES[type].BYTES_PER_ELEMENT;
        const buffer = this.device.createBuffer({
            size: alignedSize(size),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        try {
            return this.Tensor.fromGpuBuffer(buffer, {
                dataType: /** @type {Tensor.GpuBufferDataTypes} */ (type),
                dims: [...dims],
                dispose: () => buffer.destroy(),
            });
        } catch (error) {
            buffer.destroy();
            throw error;
        }
    }

    /** Non-owning, independently downloadable view of a bound tensor. @param {Tensor} tensor */
    view(tensor) {
        const buffer = tensor.gpuBuffer;
        const bytes = tensor.size * TYPES[tensor.type].BYTES_PER_ELEMENT;
        return this.Tensor.fromGpuBuffer(buffer, {
            dataType: /** @type {Tensor.GpuBufferDataTypes} */ (tensor.type),
            dims: [...tensor.dims],
            download: async () => {
                const staging = this.device.createBuffer({
                    size: alignedSize(bytes),
                    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
                });
                try {
                    const encoder = this.device.createCommandEncoder();
                    encoder.copyBufferToBuffer(buffer, 0, staging, 0, staging.size);
                    this.device.queue.submit([encoder.finish()]);
                    await staging.mapAsync(GPUMapMode.READ);
                    return new TYPES[tensor.type](staging.getMappedRange().slice(0, bytes));
                } finally {
                    staging.destroy();
                }
            },
        });
    }

    /** @param {Tensor} source @param {Tensor} target */
    upload(source, target) {
        if (
            source.type !== target.type ||
            source.dims.length !== target.dims.length ||
            source.dims.some((d, i) => d !== target.dims[i])
        )
            throw new Error('Decode graph capture requires fixed input types and shapes.');
        if (source.location === 'gpu-buffer') {
            if (!(source.gpuBuffer.usage & GPUBufferUsage.COPY_SRC))
                throw new Error('GPU inputs require GPUBufferUsage.COPY_SRC.');
            const encoder = this.device.createCommandEncoder();
            encoder.copyBufferToBuffer(
                source.gpuBuffer,
                0,
                target.gpuBuffer,
                0,
                Math.ceil((source.size * TYPES[source.type].BYTES_PER_ELEMENT) / 4) * 4,
            );
            this.device.queue.submit([encoder.finish()]);
        } else {
            const data = /** @type {Exclude<Tensor['data'], string[]>} */ (source.data);
            let bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
            if (bytes.byteLength % 4) {
                const padded = new Uint8Array(Math.ceil(bytes.byteLength / 4) * 4);
                padded.set(bytes);
                bytes = padded;
            }
            this.device.queue.writeBuffer(target.gpuBuffer, 0, /** @type {Uint8Array<ArrayBuffer>} */ (bytes));
        }
    }

    /** Import a compact external cache into each head's fixed-capacity slot. */
    importCache(source, target, length) {
        const bytes = TYPES[source.type].BYTES_PER_ELEMENT;
        if (source.type !== target.type || source.dims[1] !== target.dims[1] || source.dims[3] !== target.dims[3])
            throw new Error('External KV cache shape/type does not match this decoder.');
        const rowBytes = length * source.dims[3] * bytes;
        const sourceStride = source.dims[2] * source.dims[3] * bytes;
        const targetStride = target.dims[2] * target.dims[3] * bytes;
        if (source.location === 'gpu-buffer') {
            if (!(source.gpuBuffer.usage & GPUBufferUsage.COPY_SRC))
                throw new Error('GPU KV inputs require GPUBufferUsage.COPY_SRC.');
            const encoder = this.device.createCommandEncoder();
            for (let head = 0; head < source.dims[1]; ++head)
                encoder.copyBufferToBuffer(
                    source.gpuBuffer,
                    head * sourceStride,
                    target.gpuBuffer,
                    head * targetStride,
                    rowBytes,
                );
            this.device.queue.submit([encoder.finish()]);
        } else {
            for (let head = 0; head < source.dims[1]; ++head) {
                const data = new Uint8Array(source.data.buffer, source.data.byteOffset + head * sourceStride, rowBytes);
                this.device.queue.writeBuffer(target.gpuBuffer, head * targetStride, data);
            }
        }
    }

    /** @param {Map<string, Tensor>} tensors */
    clearTensors(tensors) {
        for (const tensor of tensors.values()) tensor.dispose();
        tensors.clear();
    }

    async dispose() {
        this.disposed = true;
        this.clearTensors(this.cache);
        this.clearTensors(this.decodeInputs);
        this.clearTensors(this.decodeOutputs);
    }
}
