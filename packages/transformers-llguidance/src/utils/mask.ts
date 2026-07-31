import { type Tensor } from '@huggingface/transformers';
import type { LLGuidanceMaskResult } from 'llguidance';

type LogitsData = Float32Array | Float64Array | number[];

export function summarizeMaskResult(result: LLGuidanceMaskResult):
    | { stop: true; reason: 'accepted' | 'dead_end' }
    | {
          maskLength: number;
          vocabSize: number;
          allowed: number;
          sampleAllowedTokenIds: number[];
      } {
    if (!('mask' in result)) {
        return { stop: true, reason: result.reason };
    }

    return {
        maskLength: result.mask.length,
        vocabSize: result.vocabSize,
        allowed: countAllowed(result.mask, result.vocabSize),
        sampleAllowedTokenIds: sampleAllowedTokenIds(result.mask, result.vocabSize),
    };
}

export function applyMask(logits: Tensor, mask: Uint32Array, vocabSize: number, includeSummary = false) {
    if (!Number.isInteger(vocabSize) || vocabSize <= 0) {
        throw new Error(`llguidance returned an invalid vocabulary size: ${vocabSize}.`);
    }

    const data = logits.data as LogitsData;
    const stride = logits.dims.at(-1);
    if (stride === undefined || !Number.isInteger(stride) || stride <= 0) {
        throw new Error('LlguidanceConstraint requires logits with a vocabulary dimension.');
    }
    if (vocabSize > stride) {
        throw new Error(`llguidance vocabulary size ${vocabSize} exceeds logits vocabulary size ${stride}.`);
    }
    if (mask.length < Math.ceil(vocabSize / 32)) {
        throw new Error(`llguidance returned a mask that is too short for vocabulary size ${vocabSize}.`);
    }

    const batchSize = Math.floor(data.length / stride);
    for (let batch = 0; batch < batchSize; ++batch) {
        const offset = batch * stride;
        applyPackedMask(data, mask, offset, vocabSize);
        if (stride > vocabSize) {
            // Logits padded beyond the grammar vocab can never be committed.
            data.fill(-Infinity, offset + vocabSize, offset + stride);
        }
    }

    if (!includeSummary) return undefined;

    const allowed = countAllowed(mask, vocabSize);
    return { vocabSize, batchSize, masked: (vocabSize - allowed) * batchSize, allowed };
}

export function forceTokens(logits: Tensor, tokenIds: number[]) {
    const data = logits.data as LogitsData;
    const vocabSize = logits.dims.at(-1);
    if (vocabSize === undefined || !Number.isInteger(vocabSize) || vocabSize <= 0) {
        throw new Error('LlguidanceConstraint requires logits with a vocabulary dimension.');
    }
    if (tokenIds.length === 0) {
        throw new Error('LlguidanceConstraint cannot stop on acceptance because the tokenizer has no EOS token ID.');
    }

    const kept = tokenIds.map((tokenId) => {
        if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= vocabSize) {
            throw new Error(`Tokenizer EOS token ID ${tokenId} is outside logits vocabulary size ${vocabSize}.`);
        }
        return data[tokenId];
    });

    data.fill(-Infinity);
    for (let i = 0; i < tokenIds.length; ++i) {
        data[tokenIds[i]] = Number.isFinite(kept[i]) ? kept[i] : 0;
    }
}

// Hot path: runs once per generated token over the whole vocab. Grammar masks
// are skewed: most 32-token words are either fully allowed (skip) or fully
// banned (memset), so per-bit work only happens on the few mixed words.
function applyPackedMask(data: LogitsData, mask: Uint32Array, offset: number, vocabSize: number) {
    const numWords = vocabSize >>> 5;
    for (let word = 0; word < numWords; ++word) {
        const bits = mask[word] | 0;
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
        if (!(mask[tokenId >>> 5] & (1 << (tokenId & 31)))) {
            data[offset + tokenId] = -Infinity;
        }
    }
}

function countAllowed(mask: Uint32Array, vocabSize: number) {
    let allowed = 0;
    for (let tokenId = 0; tokenId < vocabSize; ++tokenId) {
        if (isAllowed(mask, tokenId)) allowed++;
    }
    return allowed;
}

function sampleAllowedTokenIds(mask: Uint32Array, vocabSize: number) {
    const tokenIds: number[] = [];
    for (let tokenId = 0; tokenId < vocabSize && tokenIds.length < 25; ++tokenId) {
        if (isAllowed(mask, tokenId)) tokenIds.push(tokenId);
    }
    return tokenIds;
}

function isAllowed(mask: Uint32Array, tokenId: number) {
    return Boolean(mask[tokenId >> 5] & (1 << (tokenId & 31)));
}
