const HASH_ALGORITHM = 'SHA-256';

/**
 * Name of the Cache API bucket used to persist the url→hash mapping.
 * Kept separate from the main `transformers-cache` bucket so it can be
 * managed (inspected / cleared) independently.
 */
const HASH_CACHE_NAME = 'experimental_transformers-hash-cache';

/**
 * A cache implementation backed by the experimental `navigator.crossOriginStorage` API,
 * which allows sharing cached files (identified by content hash) across origins.
 *
 * Implements {@link import('../cache.js').CacheInterface}.
 *
 * @see https://github.com/explainers-by-googlers/cross-origin-storage
 */
export class CrossOriginStorage {
    /**
     * @param {import('../cache.js').CacheInterface|null} [fallbackCache]
     *   An optional fallback cache (e.g. a browser `Cache` opened via `caches.open()`) that is
     *   consulted when no file hash can be resolved for a given request.  When provided, both
     *   `match` and `put` delegate to it as a fallback/secondary store.
     */
    constructor(fallbackCache = null) {
        /** @type {import('../cache.js').CacheInterface|null} */
        this._fallbackCache = fallbackCache;
    }

    /**
     * Returns whether the `navigator.crossOriginStorage` API is available in the current environment.
     * @returns {boolean}
     */
    static isAvailable = () => typeof navigator !== 'undefined' && 'crossOriginStorage' in navigator;

    /**
     * Looks up a cached response for the given URL by resolving its SHA-256 hash and requesting
     * the corresponding file handle from cross-origin storage.
     *
     * Falls back to the `fallbackCache` (if configured) when no file hash can be resolved for
     * the request.
     *
     * Implements `CacheInterface.match`.
     *
     * @param {string} request The URL of the resource to look up.
     * @returns {Promise<Response|import('../hub/files.js').FileResponse|string|undefined>} The cached `Response`, or `undefined` if not found.
     */
    match = async (request) => {
        const hashValue = await this._getFileHash(request);
        if (!hashValue) {
            // No hash available — delegate to fallback cache if one is configured.
            if (this._fallbackCache) {
                return this._fallbackCache.match(request);
            }
            return undefined;
        }
        const hash = { algorithm: HASH_ALGORITHM, value: hashValue };
        try {
            // @ts-expect-error
            const [handle] = await navigator.crossOriginStorage.requestFileHandles([hash]);
            const blob = await handle.getFile();
            return new Response(blob);
        } catch (err) {
            // Cross-origin storage lookup failed — delegate to fallback cache if one is configured.
            if (this._fallbackCache) {
                return this._fallbackCache.match(request);
            }
            return undefined;
        }
    };

    /**
     * Stores a response in cross-origin storage, keyed by the SHA-256 hash of its body.
     *
     * When a `fallbackCache` is configured, also stores the response there so that subsequent
     * requests that cannot resolve a hash still have a warm entry.
     *
     * Implements `CacheInterface.put`.
     *
     * @param {string} request The URL of the resource (used to derive a cache key).
     * @param {Response} response The response whose body will be written to the cache.
     * @returns {Promise<void>}
     */
    put = async (request, response) => {
        const blob = await response.blob();
        const hash = await this._getBlobHash(blob);
        // @ts-expect-error
        const [handle] = await navigator.crossOriginStorage.requestFileHandles([hash], { create: true });
        const writableStream = await handle.createWritable();
        await writableStream.write(blob);
        await writableStream.close();

        // Populate the fallback cache as well so that a future miss (e.g. hash unavailable)
        // can still be served without a full network round-trip.
        if (this._fallbackCache) {
            try {
                await this._fallbackCache.put(request, new Response(blob));
            } catch {
                // Fallback cache write failure is non-fatal.
            }
        }
    };

    /**
     * Resolves the SHA-256 hash for a Hugging Face resource URL by reading its raw Git LFS
     * pointer file. Uses a network-first strategy: always attempts a live fetch and persists
     * the result to the Cache API. Falls back to the cached hash when the network is
     * unavailable, so cross-origin storage lookups continue to work offline.
     *
     * Supports any `/resolve/<ref>/` URL (not limited to `/resolve/main/onnx/`).
     *
     * @see https://huggingface.co/docs/hub/en/storage-backends#xet
     * @param {string} url The resolved Hugging Face URL of the resource.
     * @returns {Promise<string|null>} The hex-encoded SHA-256 hash, or `null` if unavailable.
     */
    _getFileHash = async (url) => {
        if (!/\/resolve\//.test(url)) {
            return null;
        }

        const rawUrl = url.replace(/\/resolve\//, '/raw/');

        try {
            // Network-first: fetch the LFS pointer file and cache the hash for offline use.
            const text = await fetch(rawUrl).then((response) => response.text());
            if (!text.includes('oid sha256:')) {
                return null;
            }
            const hash = text.replace(/.*?\n^oid sha256:(\w+)\n.*?$/gm, '$1') || null;
            if (hash) {
                try {
                    const hashCache = await caches.open(HASH_CACHE_NAME);
                    await hashCache.put(rawUrl, new Response(hash));
                } catch {
                    // Cache API unavailable (e.g. non-secure context): hash still returned.
                }
            }
            return hash;
        } catch {
            // Network unavailable: fall back to the last cached hash so offline lookups work.
            try {
                const hashCache = await caches.open(HASH_CACHE_NAME);
                const cached = await hashCache.match(rawUrl);
                if (cached) {
                    return cached.text();
                }
            } catch {
                // Cache API also unavailable_ nothing we can do.
            }
            return null;
        }
    };

    /**
     * Computes the SHA-256 hash of a `Blob`'s contents.
     *
     * @param {Blob} blob The blob to hash.
     * @returns {Promise<{algorithm: string, value: string}>} An object containing the algorithm
     *   identifier (`"SHA-256"`) and the lowercase hex-encoded hash value.
     */
    _getBlobHash = async (blob) => {
        const arrayBuffer = await blob.arrayBuffer();
        const hashBuffer = await crypto.subtle.digest(HASH_ALGORITHM, arrayBuffer);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        const hashHex = hashArray.map((byte) => byte.toString(16).padStart(2, '0')).join('');

        return {
            algorithm: HASH_ALGORITHM,
            value: hashHex,
        };
    };
}
