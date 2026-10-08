function reshape(data, dims, offset = 0) {
    if (dims.length === 1) return Array.from(data.slice(offset, offset + dims[0]));
    const stride = dims.slice(1).reduce((a, b) => a * b, 1);
    return Array.from({ length: dims[0] }, (_, i) => reshape(data, dims.slice(1), offset + i * stride));
}

// Tokenization only needs a small host-side tensor for its default return type.
// Keeping it separate avoids importing the ONNX-backed inference Tensor.
export class Tensor {
    constructor(type, data, dims) {
        this.type = type;
        this.data = data;
        this.dims = dims;
        this.size = data.length;
    }

    tolist() {
        return reshape(this.data, this.dims);
    }
}
