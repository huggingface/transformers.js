
/**
 * A cache class that stores past key values as named tensors.
 */
export class DynamicCache {
    /**
     * Create a DynamicCache, optionally pre-populated with entries.
     * @param {Record<string, import('./utils/tensor.js').Tensor>} [entries] Initial name→Tensor mappings.
     */
    constructor(entries) {
        if (entries) {
            Object.assign(this, entries);
        }
    }

    /**
     * Get the cached sequence length.
     * For hybrid models (e.g., Qwen3.5), finds a standard `past_key_values.*`
     * entry to determine the true past length.
     * @returns {number} The past sequence length.
     */
    get_seq_length() {
        for (const name in this) {
            if (name.startsWith('past_key_values.')) {
                return /** @type {import('./utils/tensor.js').Tensor} */ (/** @type {unknown} */ (this[name])).dims.at(-2);
            }
        }
        // Fallback for non-hybrid models (all entries are attention KV)
        return /** @type {import('./utils/tensor.js').Tensor} */ (Object.values(this)[0]).dims.at(-2);
    }

    /**
     * Dispose all contained tensors whose data resides on the GPU.
     */
    dispose() {
        for (const t of /** @type {import('./utils/tensor.js').Tensor[]} */ (Object.values(this))) {
            if (t.location === 'gpu-buffer') {
                t.dispose();
            }
        }
    }
}
