import { compileJsonSchema } from './json';
import { compileRegex } from './regex';
import { extractTokenizer, type TokenizerData } from './tokenizer';
import type { ConstraintState, JSONSchema, TokenizerSource } from './types';

type TrieNode = { childBytes: number[]; childNodes: TrieNode[]; tokenIds: number[] };
type CachedTokenizer = {
    data: TokenizerData;
    trie: TrieNode;
    whitespaceTokenIds: number[];
    stringExceptionalTrie: TrieNode;
    stringSafeMask: Uint32Array;
    stringSafeCount: number;
    stringSafeLengths: Uint32Array;
    maxStringSafeLength: number;
    maxTokenByteLength: number;
    boundedStringMasks: Map<number, { mask: Uint32Array; count: number }>;
    schemaMaskCaches: WeakMap<object, MaskCache>;
    booleanSchemaMaskCaches: [MaskCache, MaskCache];
    jsonObjectMaskCache: MaskCache;
    regexMaskCaches: Map<string, MaskCache>;
};
type ResponseFormat =
    | { type: 'json_object' }
    | { type: 'json_schema'; json_schema: JSONSchema }
    | { type: 'regex'; regex: string };

export type TokenConstraint = {
    vocabSize: number;
    fillMask(target: Uint32Array): boolean;
    commit(tokenId: number): boolean;
    repeatedWhitespace(): { tokenIds: readonly number[]; count: number } | undefined;
};

const tokenizerCache = new WeakMap<object, CachedTokenizer>();
const JSON_OBJECT_SCHEMA: JSONSchema = { type: 'object' };

/**
 * Builds and caches the tokenizer-derived data structures (token tries, string
 * masks) ahead of time. This is the expensive part of creating the first
 * constraint for a tokenizer (hundreds of milliseconds for a 256k vocabulary),
 * so calling this right after loading a model moves that cost off the first
 * generation. Subsequent calls with the same tokenizer are free.
 */
export function prepareTokenizer(tokenizerSource: TokenizerSource): void {
    cachedTokenizer(tokenizerSource);
}

export function createTokenConstraint(
    tokenizerSource: TokenizerSource,
    responseFormat: ResponseFormat,
): TokenConstraint {
    const tokenizer = cachedTokenizer(tokenizerSource);
    const machine = createMachine(responseFormat, tokenizer);
    const maskCache = cacheFor(tokenizer, responseFormat);
    let state = machine.initial;
    // Post-transition states discovered during the trie walk, so commit() can
    // reuse them. Entries are only valid when their stamp matches the current
    // fillMask() generation; bumping the stamp invalidates all of them at once
    // without refilling the vocabulary-sized array on every step.
    const tokenStates: Array<unknown> = new Array(tokenizer.data.tokens.length);
    const tokenStamps = new Int32Array(tokenizer.data.tokens.length);
    let stamp = 0;
    let consecutiveWhitespace = 0;
    const tracksJsonWhitespace = responseFormat.type !== 'regex';

    return {
        vocabSize: tokenizer.data.tokens.length,
        fillMask(target) {
            const words = Math.ceil(tokenizer.data.tokens.length / 32);
            if (target.length < words) throw new RangeError(`Mask target requires at least ${words} words.`);
            target.fill(0);
            stamp++;
            const cacheKey = machine.maskKey?.(state);
            const cachedMask = cacheKey === undefined ? undefined : maskCache?.get(cacheKey);
            if (cachedMask !== undefined) {
                target.set(cachedMask);
                return true;
            }
            let allowed = 0;
            if (machine.accepting(state)) {
                setBit(target, tokenizer.data.eosTokenId);
                allowed++;
            }
            const stringCapacity = machine.stringCapacity?.(state);
            if (stringCapacity !== undefined) {
                const safe = boundedStringMask(tokenizer, stringCapacity);
                target.set(safe.mask);
                allowed += safe.count;
            }
            const nodes: TrieNode[] = [stringCapacity === undefined ? tokenizer.trie : tokenizer.stringExceptionalTrie];
            const states: unknown[] = [state];
            while (nodes.length > 0) {
                const node = nodes.pop()!;
                const current = states.pop()!;
                for (const tokenId of node.tokenIds) {
                    if (tokenizer.data.specialTokenIds.has(tokenId)) continue;
                    setBit(target, tokenId);
                    tokenStates[tokenId] = current;
                    tokenStamps[tokenId] = stamp;
                    allowed++;
                }
                for (let index = 0; index < node.childNodes.length; ++index) {
                    const next = machine.transition(current, node.childBytes[index]);
                    if (!machine.viable(next)) continue;
                    nodes.push(node.childNodes[index]);
                    states.push(next);
                }
            }
            if (allowed > 0 && cacheKey !== undefined) {
                maskCache?.set(cacheKey, target.subarray(0, words));
            }
            return allowed > 0;
        },
        commit(tokenId) {
            if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= tokenizer.data.tokens.length) {
                throw new RangeError(`Token ${tokenId} is outside the tokenizer vocabulary.`);
            }
            if (tokenId === tokenizer.data.eosTokenId) {
                if (!machine.accepting(state)) throw new Error(`Token ${tokenId} does not satisfy the constraint.`);
                return true;
            }
            if (tokenizer.data.specialTokenIds.has(tokenId)) {
                throw new Error(`Token ${tokenId} does not satisfy the constraint.`);
            }
            let next: unknown;
            if (tokenStamps[tokenId] === stamp && stamp > 0) {
                next = tokenStates[tokenId];
            } else {
                next = state;
                for (const byte of tokenizer.data.tokens[tokenId]) next = machine.transition(next, byte);
            }
            stamp++;
            if (!machine.viable(next)) throw new Error(`Token ${tokenId} does not satisfy the constraint.`);
            consecutiveWhitespace =
                tracksJsonWhitespace && next === state && isJsonWhitespace(tokenizer.data.tokens[tokenId])
                    ? consecutiveWhitespace + 1
                    : 0;
            state = next;
            return false;
        },
        repeatedWhitespace() {
            if (consecutiveWhitespace === 0) return undefined;
            return { tokenIds: tokenizer.whitespaceTokenIds, count: consecutiveWhitespace };
        },
    };
}

function createMachine(responseFormat: ResponseFormat, tokenizer: CachedTokenizer): ConstraintState<unknown> {
    if (responseFormat?.type === 'regex') {
        if (typeof responseFormat.regex !== 'string') throw new TypeError('response_format.regex must be a string.');
        return compileRegex(responseFormat.regex) as ConstraintState<unknown>;
    }
    if (responseFormat?.type === 'json_schema') {
        return compileJsonSchema(responseFormat.json_schema, tokenizer.maxTokenByteLength) as ConstraintState<unknown>;
    }
    if (responseFormat?.type === 'json_object') {
        return compileJsonSchema(JSON_OBJECT_SCHEMA, tokenizer.maxTokenByteLength) as ConstraintState<unknown>;
    }
    throw new TypeError(`Unsupported response format: ${String((responseFormat as { type?: unknown })?.type)}.`);
}

function cachedTokenizer(source: TokenizerSource): CachedTokenizer {
    let cached = tokenizerCache.get(source as object);
    if (cached === undefined) {
        const data = extractTokenizer(source);
        const stringExceptionalTokenIds: number[] = [];
        const whitespaceTokenIds: number[] = [];
        const stringSafeMask = new Uint32Array(Math.ceil(data.tokens.length / 32));
        const stringSafeLengths = new Uint32Array(data.tokens.length);
        let stringSafeCount = 0;
        let maxStringSafeLength = 0;
        let maxTokenByteLength = 0;
        for (let tokenId = 0; tokenId < data.tokens.length; ++tokenId) {
            const special = data.specialTokenIds.has(tokenId);
            if (!special && isJsonWhitespace(data.tokens[tokenId])) whitespaceTokenIds.push(tokenId);
            if (!special && data.tokens[tokenId].length > maxTokenByteLength) {
                maxTokenByteLength = data.tokens[tokenId].length;
            }
            const length = special ? undefined : safeStringTokenLength(data.tokens[tokenId]);
            if (length !== undefined) {
                stringSafeCount++;
                stringSafeLengths[tokenId] = length;
                if (length > maxStringSafeLength) maxStringSafeLength = length;
                setBit(stringSafeMask, tokenId);
            } else {
                stringExceptionalTokenIds.push(tokenId);
            }
        }
        cached = {
            data,
            trie: createTrie(data.tokens),
            whitespaceTokenIds,
            stringExceptionalTrie: createTrie(data.tokens, stringExceptionalTokenIds),
            stringSafeMask,
            stringSafeCount,
            stringSafeLengths,
            maxStringSafeLength,
            maxTokenByteLength,
            boundedStringMasks: new Map(),
            schemaMaskCaches: new WeakMap(),
            booleanSchemaMaskCaches: [new MaskCache(), new MaskCache()],
            jsonObjectMaskCache: new MaskCache(),
            regexMaskCaches: new Map(),
        };
        tokenizerCache.set(source as object, cached);
    }
    return cached;
}

function cacheFor(tokenizer: CachedTokenizer, responseFormat: ResponseFormat): MaskCache | undefined {
    if (responseFormat.type === 'regex') {
        let cache = tokenizer.regexMaskCaches.get(responseFormat.regex);
        if (cache === undefined) {
            cache = new MaskCache();
            if (tokenizer.regexMaskCaches.size >= 16) {
                tokenizer.regexMaskCaches.delete(tokenizer.regexMaskCaches.keys().next().value!);
            }
            tokenizer.regexMaskCaches.set(responseFormat.regex, cache);
        }
        return cache;
    }
    if (responseFormat.type === 'json_object') return tokenizer.jsonObjectMaskCache;
    const schema = responseFormat.json_schema;
    if (typeof schema === 'boolean') return tokenizer.booleanSchemaMaskCaches[schema ? 1 : 0];
    let cache = tokenizer.schemaMaskCaches.get(schema);
    if (cache === undefined) {
        cache = new MaskCache();
        tokenizer.schemaMaskCaches.set(schema, cache);
    }
    return cache;
}

class MaskCache {
    private readonly masks = new Map<string, Uint32Array>();
    private words = 0;

    get(key: string): Uint32Array | undefined {
        const mask = this.masks.get(key);
        if (mask === undefined) return undefined;
        this.masks.delete(key);
        this.masks.set(key, mask);
        return mask;
    }

    set(key: string, source: Uint32Array): void {
        const mask = source.slice();
        const previous = this.masks.get(key);
        if (previous !== undefined) {
            this.words -= previous.length;
            this.masks.delete(key);
        }
        this.masks.set(key, mask);
        this.words += mask.length;
        while (this.masks.size > 256 || this.words > 1_048_576) {
            const oldestKey = this.masks.keys().next().value!;
            const oldest = this.masks.get(oldestKey)!;
            this.masks.delete(oldestKey);
            this.words -= oldest.length;
        }
    }
}

function createTrie(tokens: Uint8Array[], tokenIds?: number[]): TrieNode {
    const root: TrieNode = { childBytes: [], childNodes: [], tokenIds: [] };
    const size = tokenIds === undefined ? tokens.length : tokenIds.length;
    for (let index = 0; index < size; ++index) {
        const tokenId = tokenIds === undefined ? index : tokenIds[index];
        const bytes = tokens[tokenId];
        let node = root;
        for (let position = 0; position < bytes.length; ++position) {
            const byte = bytes[position];
            const childIndex = node.childBytes.indexOf(byte);
            if (childIndex === -1) {
                const child: TrieNode = { childBytes: [], childNodes: [], tokenIds: [] };
                node.childBytes.push(byte);
                node.childNodes.push(child);
                node = child;
            } else {
                node = node.childNodes[childIndex];
            }
        }
        node.tokenIds.push(tokenId);
    }
    return root;
}

const safeStringDecoder = new TextDecoder('utf-8', { fatal: true });

function safeStringTokenLength(bytes: Uint8Array): number | undefined {
    if (bytes.length === 0) return undefined;
    let length = 0;
    for (const byte of bytes) {
        if (byte < 0x20 || byte === 0x22 || byte === 0x5c) return undefined;
        // Code points equal non-continuation bytes in valid UTF-8.
        if ((byte & 0xc0) !== 0x80) length++;
    }
    try {
        safeStringDecoder.decode(bytes);
    } catch {
        return undefined;
    }
    return length;
}

function boundedStringMask(tokenizer: CachedTokenizer, capacity: number): { mask: Uint32Array; count: number } {
    if (capacity >= tokenizer.maxStringSafeLength) {
        return { mask: tokenizer.stringSafeMask, count: tokenizer.stringSafeCount };
    }
    let cached = tokenizer.boundedStringMasks.get(capacity);
    if (cached !== undefined) return cached;
    const mask = new Uint32Array(Math.ceil(tokenizer.data.tokens.length / 32));
    let count = 0;
    for (let tokenId = 0; tokenId < tokenizer.stringSafeLengths.length; ++tokenId) {
        const length = tokenizer.stringSafeLengths[tokenId];
        if (length === 0 || length > capacity) continue;
        setBit(mask, tokenId);
        count++;
    }
    cached = { mask, count };
    tokenizer.boundedStringMasks.set(capacity, cached);
    return cached;
}

function isJsonWhitespace(bytes: Uint8Array): boolean {
    return bytes.length > 0 && bytes.every((byte) => byte === 0x09 || byte === 0x0a || byte === 0x0d || byte === 0x20);
}

function setBit(mask: Uint32Array, tokenId: number): void {
    mask[tokenId >>> 5] |= 1 << (tokenId & 31);
}
