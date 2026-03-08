/**
 * A cache class that stores past key values as named tensors.
 */
export class DynamicCache extends /** @type {new () => Record<string, any>} */ (/** @type {unknown} */ (Object)) {
    /**
     * Create a DynamicCache, optionally pre-populated with entries.
     * @param {Record<string, import('./utils/tensor.js').Tensor>} [entries] Initial name→Tensor mappings.
     */
    constructor(entries) {
        super();
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
                return this[name].dims.at(-2);
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
        for (const t of Object.values(this)) {
            if (t.location === 'gpu-buffer') {
                promises.push(t.dispose());
            }
        }
        await Promise.all(promises);
    }
}
