import { type Tensor } from '@huggingface/transformers';
import { type GuidanceMask, type GuidanceMaskResult } from './types';

type LogitsData = Float32Array | Float64Array | number[];

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

export function applyMask(logits: Tensor, mask: GuidanceMask, vocabSize?: number, includeSummary = false) {
    if (!vocabSize) {
        return includeSummary ? { vocabSize, batchSize: 0, masked: 0, allowed: undefined } : undefined;
    }

    const data = logits.data as LogitsData;
    const stride = (logits.dims?.at?.(-1) as number) || vocabSize;
    const bound = Math.min(vocabSize, stride);
    const batchSize = Math.max(1, Math.floor(data.length / stride));
    const packed = mask.length < vocabSize;

    for (let batch = 0; batch < batchSize; ++batch) {
        const offset = batch * stride;
        if (packed) {
            applyPackedMask(data, mask, offset, bound);
        } else {
            for (let tokenId = 0; tokenId < bound; ++tokenId) {
                if (!mask[tokenId]) data[offset + tokenId] = -Infinity;
            }
        }
        if (stride > vocabSize) {
            // Logits padded beyond the grammar vocab can never be committed.
            data.fill(-Infinity, offset + vocabSize, offset + stride);
        }
    }

    if (!includeSummary) return undefined;

    const allowed = countAllowed(mask, vocabSize) ?? 0;
    return { vocabSize, batchSize, masked: (vocabSize - allowed) * batchSize, allowed };
}

// Forces a single token by banning everything else, e.g. for fast-forward splices.
export function forceToken(logits: Tensor, tokenId: number, vocabSize?: number) {
    if (!vocabSize) return;

    const data = logits.data as LogitsData;
    const stride = (logits.dims?.at?.(-1) as number) || vocabSize;
    const batchSize = Math.max(1, Math.floor(data.length / stride));

    for (let batch = 0; batch < batchSize; ++batch) {
        const offset = batch * stride;
        const kept = data[offset + tokenId];
        data.fill(-Infinity, offset, offset + stride);
        data[offset + tokenId] = Number.isFinite(kept) ? kept : 0;
    }
}

// Hot path: runs once per generated token over the whole vocab. Grammar masks
// are skewed — most 32-token words are either fully allowed (skip) or fully
// banned (memset) — so per-bit work only happens on the few mixed words.
function applyPackedMask(data: LogitsData, mask: GuidanceMask, offset: number, vocabSize: number) {
    const numWords = vocabSize >>> 5;
    for (let word = 0; word < numWords; ++word) {
        const bits = (mask[word] as number) | 0;
        if (bits === -1) continue;
        const base = offset + (word << 5);
        if (bits === 0) {
            data.fill(-Infinity, base, base + 32);
            continue;
        }
        for (let bit = 0; bit < 32; ++bit) {
            if (!(bits & (1 << bit))) data[base + bit] = -Infinity;
        }
    }

    for (let tokenId = numWords << 5; tokenId < vocabSize; ++tokenId) {
        if (!((mask[tokenId >>> 5] as number) & (1 << (tokenId & 31)))) {
            data[offset + tokenId] = -Infinity;
        }
    }
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
