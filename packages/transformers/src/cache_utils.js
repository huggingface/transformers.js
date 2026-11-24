/**
 * @file Cache classes used by `generate()` to preserve and reuse decoder
 * past key/value tensors across calls. Pass a `DynamicCache` through
 * generation options when you need to persist KV state between turns.
 *
 * @module generation/cache
 */

import { Tensor } from './utils/tensor.js';

const GPU_BUFFER_UNSUPPORTED =
    'DynamicCache operations that copy or slice cache tensors are not yet supported for gpu-buffer tensors.';

/**
 * @param {Tensor} tensor
 * @param {string} operation
 */
function assertCpuTensor(tensor, operation) {
    if (tensor.location === 'gpu-buffer') {
        throw new Error(`${GPU_BUFFER_UNSUPPORTED} (during ${operation})`);
    }
}

/**
 * @param {string} name
 * @returns {boolean}
 */
function isDecoderSelfAttentionKey(name) {
    return name.startsWith('past_key_values.') && !name.includes('.encoder.');
}

/**
 * @param {string} name
 * @returns {boolean}
 */
function isNonAttentionStateKey(name) {
    return !name.startsWith('past_key_values.');
}

/**
 * @param {number} maxLength
 * @param {number} currentLength
 * @returns {number}
 */
function resolveTargetLength(maxLength, currentLength) {
    if (maxLength >= 0) {
        return maxLength;
    }
    const remove = -maxLength;
    if (remove > currentLength) {
        throw new RangeError(`Cannot remove ${remove} positions from a cache of length ${currentLength}.`);
    }
    return currentLength - remove;
}

/**
 * Build slice arguments to keep the first `targetLength` positions along the sequence axis.
 * @param {number[]} dims
 * @param {number} targetLength
 * @returns {(number|number[]|null)[]}
 */
function sliceArgsForSeqLen(dims, targetLength) {
    const seqAxis = dims.length - 2;
    /** @type {(number|number[]|null)[]} */
    const args = [];
    for (let i = 0; i < dims.length; ++i) {
        if (i === seqAxis) {
            args.push([0, targetLength]);
        } else {
            args.push(null);
        }
    }
    return args;
}

/**
 * @param {Record<string, Tensor>} self
 * @returns {string[]}
 */
function listNonAttentionKeys(self) {
    /** @type {string[]} */
    const keys = [];
    for (const name in self) {
        if (isNonAttentionStateKey(name)) {
            keys.push(name);
        }
    }
    return keys;
}

/**
 * @param {Record<string, Tensor>} self
 * @returns {number}
 */
function getDecoderAttentionSeqLength(self) {
    let length = null;
    for (const name in self) {
        if (!isDecoderSelfAttentionKey(name)) continue;
        const seqLen = self[name].dims.at(-2);
        if (length === null) {
            length = seqLen;
        } else if (seqLen !== length) {
            throw new Error(
                `Cannot crop: decoder attention cache entry "${name}" has sequence length ${seqLen}, expected ${length}. ` +
                    'Sliding-window or mismatched layers require a different strategy.',
            );
        }
    }
    if (length === null) {
        throw new Error('Unable to determine sequence length from the cache.');
    }
    return length;
}

/**
 * @param {Record<string, Tensor>} self
 * @param {number} targetLength
 * @param {boolean} allowNonAttention If false, throws when non-attention keys are present (public crop).
 * @param {number} [currentLength] Decoder attention length when already known (avoids a second scan from {@link _DynamicCache#crop}).
 */
function cropDecoderAttentionInPlace(self, targetLength, allowNonAttention, currentLength) {
    if (!allowNonAttention) {
        const extra = listNonAttentionKeys(self);
        if (extra.length > 0) {
            throw new Error(
                `Cannot crop a hybrid cache with non-attention state (${extra.join(', ')}). ` +
                    'Use snapshot() and restore() to save and restore conv/recurrent state, or crop only attention-only caches.',
            );
        }
    }

    const length = currentLength ?? getDecoderAttentionSeqLength(self);
    if (targetLength >= length) {
        return;
    }

    /** @type {Record<string, Tensor>} */
    const replacements = Object.create(null);
    for (const name in self) {
        if (!isDecoderSelfAttentionKey(name)) continue;
        const tensor = self[name];
        assertCpuTensor(tensor, `crop("${name}")`);
        const sliced = tensor.slice(...sliceArgsForSeqLen(tensor.dims, targetLength));
        replacements[name] = sliced;
    }

    /** @type {_DynamicCache} */ (/** @type {unknown} */ (self)).update(replacements);
}

/**
 * @param {Tensor} tensor
 * @returns {Tensor}
 */
function deepCloneTensor(tensor) {
    assertCpuTensor(tensor, 'clone');
    return tensor.clone();
}

/**
 * A cache class that stores past key values as named tensors.
 */
class _DynamicCache {
    /**
     * Create a DynamicCache, optionally pre-populated with entries.
     * @param {Record<string, Tensor>} [entries] Initial name→Tensor mappings.
     */
    constructor(entries) {
        if (!entries) return;
        for (const key in entries) {
            if (key in this) {
                throw new TypeError(`Key "${key}" conflicts with an existing property on DynamicCache`);
            }
            const value = entries[key];
            if (!(value instanceof Tensor)) {
                throw new TypeError(`Expected a Tensor for key "${key}", got ${typeof value}`);
            }
            this[key] = value;
        }
    }

    /**
     * Get the cached sequence length. This requires at least one attention cache entry to be present.
     * @returns {number} The past sequence length.
     */
    get_seq_length() {
        /** @type {Record<string, Tensor>} */
        const self = /** @type {any} */ (this);

        if (Object.keys(self).length === 0) {
            return 0;
        }

        for (const name in self) {
            if (name.startsWith('past_key_values.')) {
                return self[name].dims.at(-2);
            }
        }
        throw new Error('Unable to determine sequence length from the cache.');
    }

    /**
     * Trim decoder self-attention cache entries along the sequence dimension.
     *
     * - `maxLength >= 0`: keep the first `maxLength` key/value positions.
     *   {@link _DynamicCache#crop `crop(0)`} keeps zero positions and clears the decoder attention cache.
     * - `maxLength < 0`: remove `|maxLength|` positions from the end (Python-compatible).
     *
     * Encoder-side entries (`past_key_values.*.encoder.*`) are left unchanged. Caches that
     * include conv/SSM/recurrent state cannot be cropped; use {@link _DynamicCache#snapshot} and
     * {@link _DynamicCache#restore} instead.
     *
     * After `generate()`, the cache typically holds `prompt_length + generated_length - 1` tokens.
     * To reuse a shared system prefix for a new user turn, crop to the common prefix length while
     * leaving at least one prompt token to decode (e.g. `cache.crop(commonPrefixLen - 1)`).
     *
     * @param {number} maxLength Target length (non-negative) or number of trailing tokens to remove (negative).
     * @returns {this}
     * @throws {RangeError} If `maxLength` is not an integer or the negative removal exceeds the cache length.
     * @throws {Error} If the cache is hybrid, sequence lengths disagree, or tensors are on GPU.
     */
    crop(maxLength) {
        if (!Number.isInteger(maxLength)) {
            throw new RangeError(`maxLength must be an integer, got ${maxLength}.`);
        }

        /** @type {Record<string, Tensor>} */
        const self = /** @type {any} */ (this);
        if (Object.keys(self).length === 0) {
            return this;
        }

        const currentLength = getDecoderAttentionSeqLength(self);
        const targetLength = resolveTargetLength(maxLength, currentLength);
        cropDecoderAttentionInPlace(self, targetLength, false, currentLength);
        return this;
    }

    /**
     * Create an independent deep copy of all cache entries. Optional `length` crops decoder attention
     * on the copy only (same rules as {@link _DynamicCache#crop}).
     * @param {number} [length] If set, same signed semantics as {@link _DynamicCache#crop}: non-negative
     * values keep the first N positions; negative values remove `|N|` positions from the end.
     * @returns {DynamicCache}
     */
    clone(length) {
        /** @type {Record<string, Tensor>} */
        const self = /** @type {any} */ (this);
        /** @type {Record<string, Tensor>} */
        const entries = Object.create(null);
        for (const name in self) {
            entries[name] = deepCloneTensor(self[name]);
        }
        const copy = new DynamicCache(entries);
        if (length !== undefined) {
            copy.crop(length);
        }
        return copy;
    }

    /**
     * Capture the current attention sequence length and deep copies of non-attention state tensors
     * (e.g. `past_conv.*`, `past_ssm.*`). Use with {@link _DynamicCache#restore} to branch or roll back
     * after further generation.
     * @returns {CacheSnapshot}
     */
    snapshot() {
        /** @type {Record<string, Tensor>} */
        const self = /** @type {any} */ (this);
        const length = getDecoderAttentionSeqLength(self);
        /** @type {Record<string, Tensor>} */
        const states = Object.create(null);
        for (const name of listNonAttentionKeys(self)) {
            states[name] = deepCloneTensor(self[name]);
        }
        return {
            length,
            states,
            // Snapshots only hold CPU clones (deepCloneTensor throws on gpu-buffer); kept for API symmetry.
            dispose: async () => {},
        };
    }

    /**
     * Restore attention length and non-attention state from a {@link CacheSnapshot}. State tensors are
     * copied from the snapshot so it can be applied multiple times. Replaced GPU-resident cache tensors
     * are disposed.
     *
     * Throws if the cache is empty (no decoder attention entries), the same as when attention length
     * cannot be determined.
     *
     * @param {CacheSnapshot} snapshot Snapshot from {@link _DynamicCache#snapshot}.
     * @returns {this}
     */
    restore(snapshot) {
        /** @type {Record<string, Tensor>} */
        const self = /** @type {any} */ (this);
        const currentLength = getDecoderAttentionSeqLength(self);
        if (snapshot.length > currentLength) {
            throw new Error(
                `Cannot restore snapshot with length ${snapshot.length} into a cache of length ${currentLength}.`,
            );
        }

        const cacheStateKeys = listNonAttentionKeys(self).sort();
        const snapKeys = Object.keys(snapshot.states).sort();
        if (cacheStateKeys.length !== snapKeys.length || cacheStateKeys.some((k, i) => k !== snapKeys[i])) {
            throw new Error(
                'Snapshot state keys do not match the cache. Expected [' +
                    cacheStateKeys.join(', ') +
                    '], got [' +
                    snapKeys.join(', ') +
                    '].',
            );
        }

        cropDecoderAttentionInPlace(self, snapshot.length, true, currentLength);

        /** @type {Record<string, Tensor>} */
        const stateCopies = Object.create(null);
        for (const name of snapKeys) {
            stateCopies[name] = deepCloneTensor(snapshot.states[name]);
        }
        this.update(stateCopies);
        return this;
    }

    /**
     * Update the cache in-place with new entries, disposing replaced GPU tensors.
     * @param {Record<string, Tensor>} newEntries The new name → Tensor mappings.
     */
    update(newEntries) {
        for (const key in newEntries) {
            const oldValue = this[key];
            const newValue = newEntries[key];
            if (oldValue && oldValue !== newValue && oldValue.location === 'gpu-buffer') {
                oldValue.dispose();
            }
            this[key] = newValue;
        }
    }

    /**
     * Dispose all contained tensors whose data resides on the GPU.
     * Returns a promise that resolves when all disposals are complete.
     * @returns {Promise<void>} Promise that resolves when all GPU tensors are disposed.
     */
    async dispose() {
        const promises = [];
        for (const t of /** @type {Tensor[]} */ (Object.values(this))) {
            if (t.location === 'gpu-buffer') {
                promises.push(t.dispose());
            }
        }
        await Promise.all(promises);
    }
}

/**
 * Snapshot of hybrid cache non-attention state and attention length for {@link _DynamicCache#restore}.
 *
 * @typedef {Object} CacheSnapshot
 * @property {number} length Decoder attention sequence length at snapshot time.
 * @property {Record<string, Tensor>} states Deep copies of non-attention tensors (`past_conv.*`, etc.).
 * @property {() => Promise<void>} dispose Release GPU buffers held only by the snapshot copies.
 */

/**
 * Mutable cache of decoder past key/value tensors used by `generate()`.
 *
 * Pass a `DynamicCache` through generation options when you need to preserve
 * cache tensors across calls or inspect them from `return_dict_in_generate`.
 *
 * @typedef {Record<string, Tensor> & _DynamicCache} DynamicCache
 */

export const DynamicCache = /** @type {new (entries?: Record<string, Tensor>) => DynamicCache} */ (
    /** @type {unknown} */ (_DynamicCache)
);
