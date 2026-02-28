import { Tensor } from '../../utils/tensor.js';

/**
 * Create a stable hash key for audio samples, used by feature caches.
 * @param {Float32Array|Float64Array} audio
 * @param {number} [sampling_rate=16000]
 * @returns {string}
 */
export function createAudioCacheKey(audio, sampling_rate = 16000) {
    // FNV-1a 32-bit over quantized values for deterministic cross-runtime keys.
    let hash = 2166136261;
    hash ^= audio.length;
    hash = Math.imul(hash, 16777619);
    hash ^= sampling_rate;
    hash = Math.imul(hash, 16777619);

    // Sample stride hash to keep keying cheap for long audio.
    const stride = Math.max(1, Math.floor(audio.length / 4096));
    for (let i = 0; i < audio.length; i += stride) {
        const q = (audio[i] * 32768) | 0;
        hash ^= q;
        hash = Math.imul(hash, 16777619);
    }
    return `${sampling_rate}:${audio.length}:${(hash >>> 0).toString(16)}`;
}

/**
 * Lightweight LRU cache for extracted features.
 * Stores values as-is and tracks approximate memory usage.
 */
export class FeatureLRUCache {
    /**
     * @param {{max_entries?: number, max_size_mb?: number}} [options]
     */
    constructor({ max_entries = 128, max_size_mb = 64 } = {}) {
        this.max_entries = max_entries;
        this.max_size_mb = max_size_mb;
        this.cache = new Map();
        this.current_size_bytes = 0;
    }

    /**
     * @param {string} key
     * @returns {any|null}
     */
    get(key) {
        const entry = this.cache.get(key);
        if (!entry) return null;
        this.cache.delete(key);
        this.cache.set(key, entry);
        return entry.value;
    }

    /**
     * @param {string} key
     * @param {any} value
     * @returns {void}
     */
    set(key, value) {
        const existing = this.cache.get(key);
        if (existing) {
            this.current_size_bytes -= existing.size_bytes;
            this.cache.delete(key);
        }

        const size_bytes = estimateSizeBytes(value);
        this.cache.set(key, { value, size_bytes });
        this.current_size_bytes += size_bytes;
        this._evict();
    }

    clear() {
        this.cache.clear();
        this.current_size_bytes = 0;
    }

    stats() {
        return {
            entries: this.cache.size,
            size_mb: this.current_size_bytes / (1024 * 1024),
            max_entries: this.max_entries,
            max_size_mb: this.max_size_mb,
        };
    }

    _evict() {
        const max_bytes = this.max_size_mb * 1024 * 1024;
        while (this.cache.size > this.max_entries || this.current_size_bytes > max_bytes) {
            const oldest_key = this.cache.keys().next().value;
            if (oldest_key === undefined) break;
            const oldest = this.cache.get(oldest_key);
            this.cache.delete(oldest_key);
            this.current_size_bytes -= oldest?.size_bytes ?? 0;
        }
    }
}

function estimateSizeBytes(value) {
    if (value instanceof Tensor) {
        return value.data?.byteLength ?? 0;
    }
    if (value?.input_features instanceof Tensor) {
        let bytes = value.input_features.data?.byteLength ?? 0;
        if (value.attention_mask instanceof Tensor) {
            bytes += value.attention_mask.data?.byteLength ?? 0;
        }
        if (value.delta_features instanceof Tensor) {
            bytes += value.delta_features.data?.byteLength ?? 0;
        }
        if (value.delta_delta_features instanceof Tensor) {
            bytes += value.delta_delta_features.data?.byteLength ?? 0;
        }
        return bytes;
    }
    if (value?.byteLength) {
        return value.byteLength;
    }
    return 0;
}
