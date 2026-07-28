export const DATA_TYPES = Object.freeze({
    auto: 'auto',
    fp32: 'fp32',
    fp16: 'fp16',
    q8: 'q8',
    int8: 'int8',
    uint8: 'uint8',
    q4: 'q4',
    bnb4: 'bnb4',
    q4f16: 'q4f16',
    q2: 'q2',
    q2f16: 'q2f16',
    q1: 'q1',
    q1f16: 'q1f16',
});

/** @typedef {keyof typeof DATA_TYPES} DataType */

export const DataTypeMap = Object.freeze({
    float32: Float32Array,
    // @ts-ignore Limited availability of Float16Array across browsers.
    float16: typeof Float16Array !== 'undefined' ? Float16Array : Uint16Array,
    float64: Float64Array,
    string: Array,
    int8: Int8Array,
    uint8: Uint8Array,
    int16: Int16Array,
    uint16: Uint16Array,
    int32: Int32Array,
    uint32: Uint32Array,
    int64: BigInt64Array,
    uint64: BigUint64Array,
    bool: Uint8Array,
    uint4: Uint8Array,
    int4: Int8Array,
});
