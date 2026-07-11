import { type Tensor } from '@huggingface/transformers';
import { type GuidanceMask, type GuidanceMaskResult } from './types';

export function summarizeMaskResult(result: GuidanceMaskResult, vocabSize?: number) {
    if ('stop' in result && result.stop) {
        return { stop: true };
    }

    if (!('mask' in result)) {
        return result;
    }

    return {
        maskLength: result.mask.length,
        vocabSize: result.vocabSize ?? vocabSize,
        allowed: countAllowed(result.mask, result.vocabSize ?? vocabSize),
        sampleAllowedTokenIds: sampleAllowedTokenIds(result.mask, result.vocabSize ?? vocabSize),
    };
}

export function applyMask(logits: Tensor, mask: GuidanceMask, vocabSize?: number) {
    if (!vocabSize) {
        return { vocabSize, batchSize: 0, masked: 0, allowed: undefined };
    }

    const data = logits.data as Float32Array | Float64Array | number[];
    const batchSize = Math.max(1, data.length / vocabSize);
    let masked = 0;
    let allowed = 0;

    for (let batch = 0; batch < batchSize; ++batch) {
        const offset = batch * vocabSize;
        for (let tokenId = 0; tokenId < vocabSize; ++tokenId) {
            if (!isAllowed(mask, tokenId, vocabSize)) {
                data[offset + tokenId] = -Infinity;
                masked++;
            } else if (batch === 0) {
                allowed++;
            }
        }
    }

    return { vocabSize, batchSize, masked, allowed };
}

function countAllowed(mask: GuidanceMask, vocabSize?: number) {
    if (!vocabSize) return undefined;

    let allowed = 0;
    for (let tokenId = 0; tokenId < vocabSize; ++tokenId) {
        if (isAllowed(mask, tokenId, vocabSize)) allowed++;
    }
    return allowed;
}

function sampleAllowedTokenIds(mask: GuidanceMask, vocabSize?: number) {
    if (!vocabSize) return [];

    const tokenIds: number[] = [];
    for (let tokenId = 0; tokenId < vocabSize && tokenIds.length < 25; ++tokenId) {
        if (isAllowed(mask, tokenId, vocabSize)) tokenIds.push(tokenId);
    }
    return tokenIds;
}

function isAllowed(mask: GuidanceMask, tokenId: number, vocabSize: number) {
    if (mask.length >= vocabSize) {
        return Boolean(mask[tokenId]);
    }
    return Boolean(Number(mask[tokenId >> 5]) & (1 << (tokenId & 31)));
}
