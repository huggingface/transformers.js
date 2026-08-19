import { type Tensor } from '@huggingface/transformers';

type LogitsData = Float32Array | Float64Array | number[];

export function applyMask(logits: Tensor, mask: Uint32Array, vocabSize: number): void {
    const data = logits.data as LogitsData;
    const stride = logits.dims.at(-1)!;
    if (vocabSize > stride) {
        throw new Error(`Constraint vocabulary size ${vocabSize} exceeds logits vocabulary size ${stride}.`);
    }
    for (let offset = 0; offset < data.length; offset += stride) {
        const fullWords = vocabSize >>> 5;
        for (let word = 0; word < fullWords; ++word) {
            const bits = mask[word] | 0;
            if (bits === -1) continue;
            const start = offset + (word << 5);
            if (bits === 0) {
                data.fill(-Infinity, start, start + 32);
            } else {
                for (let bit = 0; bit < 32; ++bit) {
                    if (!(bits & (1 << bit))) data[start + bit] = -Infinity;
                }
            }
        }
        for (let tokenId = fullWords << 5; tokenId < vocabSize; ++tokenId) {
            if (!(mask[tokenId >>> 5] & (1 << (tokenId & 31)))) data[offset + tokenId] = -Infinity;
        }
        data.fill(-Infinity, offset + vocabSize, offset + stride);
    }
}
