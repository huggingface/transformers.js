import { Tensor } from './utils/tensor.js';

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
        for (const name in self) {
            if (name.startsWith('past_key_values.')) {
                return self[name].dims.at(-2);
            }
        }
        throw new Error('Unable to determine sequence length from the cache.');
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
 * @typedef {_DynamicCache & Record<string, Tensor>} DynamicCache
 */

export const DynamicCache = /** @type {new (entries?: Record<string, Tensor>) => DynamicCache} */ (
    /** @type {unknown} */ (_DynamicCache)
);
