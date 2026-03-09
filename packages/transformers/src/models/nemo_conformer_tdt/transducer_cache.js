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

    // Hash all quantized samples to minimize false cache hits across waveforms.
    for (let i = 0; i < audio.length; ++i) {
        const sample = Number.isFinite(audio[i]) ? audio[i] : 0;
        const q = Math.max(-32768, Math.min(32767, Math.round(sample * 32768)));
        hash ^= q;
        hash = Math.imul(hash, 16777619);
    }
    return `${sampling_rate}:${audio.length}:${(hash >>> 0).toString(16)}`;
}

/**
 * Lightweight LRU cache for extracted features.
 * Stores values as-is, owns cached tensor lifetimes, and tracks approximate memory usage.
 */
export class FeatureLRUCache {
    /**
     * @param {{max_entries?: number, max_size_mb?: number}} [options]
     */
    constructor({ max_entries = 128, max_size_mb = 64 } = {}) {
        if (!Number.isInteger(max_entries) || max_entries < 0) {
            throw new Error('FeatureLRUCache expected `max_entries` to be a non-negative integer.');
        }
        if (!Number.isFinite(max_size_mb) || max_size_mb < 0) {
            throw new Error('FeatureLRUCache expected `max_size_mb` to be a non-negative number.');
        }
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
        const entry = this._touch(key);
        if (!entry) return null;
        return entry.value;
    }

    /**
     * @param {string} key
     * @returns {{ value: any, release: () => void } | null}
     */
    acquire(key) {
        const entry = this._touch(key);
        if (!entry) return null;

        entry.borrowers += 1;
        let released = false;
        return {
            value: entry.value,
            release: () => {
                if (released) return;
                released = true;
                this._releaseEntry(entry);
            },
        };
    }

    /**
     * @param {string} key
     * @param {any} value
     * @returns {boolean} Whether the cache retained ownership of the supplied value.
     */
    set(key, value) {
        // Explicit no-cache mode: keep caller ownership of current values.
        if (this.max_entries === 0 || this.max_size_mb === 0) {
            if (this.cache.size > 0) {
                this.clear();
            }
            return false;
        }

        const max_bytes = this.max_size_mb * 1024 * 1024;
        const existing = this.cache.get(key);
        if (existing?.value === value) {
            // Refresh recency for unchanged value without invalidating caller-owned references.
            if (existing.size_bytes <= max_bytes) {
                this.cache.delete(key);
                this.cache.set(key, existing);
                return true;
            } else {
                this._deleteEntry(key, existing);
                return false;
            }
        }

        const size_bytes = estimateSizeBytes(value);
        if (size_bytes > max_bytes) {
            // Cannot fit in cache: keep caller ownership and skip caching.
            if (existing) {
                this._deleteEntry(key, existing);
            }
            return false;
        }

        if (existing) {
            this._deleteEntry(key, existing);
        }

        this.cache.set(key, {
            value,
            size_bytes,
            borrowers: 0,
            pendingDispose: false,
        });
        this.current_size_bytes += size_bytes;
        this._evict();
        return this.cache.get(key)?.value === value;
    }

    clear() {
        for (const [key, entry] of Array.from(this.cache.entries())) {
            this._deleteEntry(key, entry);
        }
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
            if (!oldest) break;
            this._deleteEntry(oldest_key, oldest);
        }
    }

    _touch(key) {
        const entry = this.cache.get(key);
        if (!entry) return null;
        this.cache.delete(key);
        this.cache.set(key, entry);
        return entry;
    }

    _deleteEntry(key, entry) {
        const current = this.cache.get(key);
        if (current !== entry) {
            return;
        }

        this.cache.delete(key);
        if (entry.borrowers > 0) {
            entry.pendingDispose = true;
        } else {
            this.current_size_bytes -= entry.size_bytes;
            disposeCachedValue(entry.value);
        }
    }

    _releaseEntry(entry) {
        if (entry.borrowers > 0) {
            entry.borrowers -= 1;
        }
        if (entry.borrowers === 0 && entry.pendingDispose) {
            entry.pendingDispose = false;
            this.current_size_bytes -= entry.size_bytes;
            disposeCachedValue(entry.value);
        }
    }
}

function tensorByteSize(tensor) {
    let byteLength = null;
    try {
        byteLength = /** @type {any} */ (tensor.data)?.byteLength ?? null;
    } catch {
        byteLength = null;
    }
    if (typeof byteLength === 'number' && byteLength >= 0) {
        return byteLength;
    }

    const bytesPerElement = {
        bool: 1,
        int8: 1,
        uint8: 1,
        int16: 2,
        uint16: 2,
        int32: 4,
        uint32: 4,
        int64: 8,
        uint64: 8,
        float16: 2,
        float32: 4,
        float64: 8,
    };
    return tensor.size * (bytesPerElement[tensor.type] ?? 4);
}

function collectCachedTensors(value, out = new Set()) {
    if (value instanceof Tensor) {
        out.add(value);
        return out;
    }
    if (value?.input_features instanceof Tensor) out.add(value.input_features);
    if (value?.attention_mask instanceof Tensor) out.add(value.attention_mask);
    if (value?.delta_features instanceof Tensor) out.add(value.delta_features);
    if (value?.delta_delta_features instanceof Tensor) out.add(value.delta_delta_features);
    return out;
}

function disposeCachedValue(value) {
    for (const tensor of collectCachedTensors(value)) {
        tensor.dispose();
    }
}

function estimateSizeBytes(value) {
    if (value instanceof Tensor) {
        return tensorByteSize(value);
    }
    if (value?.input_features instanceof Tensor) {
        let bytes = tensorByteSize(value.input_features);
        if (value.attention_mask instanceof Tensor) {
            bytes += tensorByteSize(value.attention_mask);
        }
        if (value.delta_features instanceof Tensor) {
            bytes += tensorByteSize(value.delta_features);
        }
        if (value.delta_delta_features instanceof Tensor) {
            bytes += tensorByteSize(value.delta_delta_features);
        }
        return bytes;
    }
    const byteLength = value?.byteLength;
    if (typeof byteLength === 'number' && Number.isFinite(byteLength) && byteLength >= 0) {
        return byteLength;
    }
    return 0;
}
