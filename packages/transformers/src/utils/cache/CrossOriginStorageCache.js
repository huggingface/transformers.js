/// <reference path="./cross-origin-storage.d.ts" />

const HASH_ALGORITHM = 'SHA-256';

/**
 * Name of the Cache API bucket used to persist the url→hash mapping.
 */
const HASH_CACHE_NAME = 'experimental_transformers-hash-cache';

/**
 * Builds the hash descriptor object expected by the cross-origin storage API.
 *
 * @param {string} value Hex-encoded SHA-256 hash.
 * @returns {{ algorithm: string, value: string }}
 */
const makeHashDescriptor = (value) => ({ algorithm: HASH_ALGORITHM, value });

/**
 * A cache implementation backed by the experimental `navigator.crossOriginStorage` API,
 * which allows sharing cached files (identified by content hash) across origins.
 *
 * Implements {@link import('../cache.js').CacheInterface}.
 *
 * @see https://github.com/explainers-by-googlers/cross-origin-storage
 */
export class CrossOriginStorage {
    /** @type {Promise<Cache> | null} */
    #hashCache = null;

    /**
     * Returns (and lazily opens) the hash cache, reusing the same promise across concurrent callers.
     * @returns {Promise<Cache>}
     */
    _getHashCache = () => {
        this.#hashCache ??= caches.open(HASH_CACHE_NAME);
        return this.#hashCache;
    };

    /**
     * Returns whether the `navigator.crossOriginStorage` API is available in the current environment.
     * @returns {boolean}
     */
    static isAvailable = () => typeof navigator !== 'undefined' && 'crossOriginStorage' in navigator;

    /**
     * Looks up a cached response for the given URL by resolving its SHA-256 hash and requesting
     * the corresponding file handle from cross-origin storage.
     *
     * Implements `CacheInterface.match`.
     *
     * @param {string} request The URL of the resource to look up.
     * @returns {Promise<Response|undefined>} The cached `Response`, or `undefined` if not found.
     */
    match = async (request) => {
        const hashValue = await this._getFileHash(request);
        if (!hashValue) {
            return undefined;
        }
        try {
            const [handle] = await navigator.crossOriginStorage.requestFileHandles([makeHashDescriptor(hashValue)]);
            const blob = await handle.getFile();
            return new Response(blob, {
                headers: {
                    'Content-Length': String(blob.size),
                },
            });
        } catch {
            return undefined;
        }
    };

    /**
     * Stores a response in cross-origin storage, keyed by its SHA-256 hash.
     *
     * For LFS-backed URLs the hash is resolved cheaply via `_getFileHash` (which checks
     * `HASH_CACHE_NAME` first, then falls back to fetching the Git LFS pointer file)
     * without reading the response body a second time.
     *
     * For non-LFS resources the hash is unknown upfront.  In that case the body is consumed
     * in the background: the stream is read to compute the content hash, the file is written
     * into cross-origin storage, and the computed hash is persisted to `HASH_CACHE_NAME`
     * so that future `match` calls can resolve the file without a network round-trip.
     *
     * Implements `CacheInterface.put`.
     *
     * @param {string} request The URL of the resource (used as the hash-cache key).
     * @param {Response} response The response whose body will be written to the cache.
     * @returns {Promise<void>}
     */
    put = async (request, response) => {
        const hashValue = await this._getFileHash(request);

        if (hashValue) {
            // Fast path: LFS hash already known. Consume the body and store directly.
            const blob = await response.blob();
            await this._storeBlobInCOS(blob, hashValue);
        } else {
            // Slow path: hash unknown. Process in the background so put() returns promptly.
            // The caller already holds a reference to the original response; we receive it
            // here only to buffer and hash its body.
            this._processAndStore(request, response.body);
        }
    };

    /**
     * Writes a blob into cross-origin storage using the given pre-computed hex hash string.
     *
     * @param {Blob} blob
     * @param {string} hashHex Hex-encoded SHA-256 hash of `blob`.
     * @returns {Promise<void>}
     */
    _storeBlobInCOS = async (blob, hashHex) => {
        const [handle] = await navigator.crossOriginStorage.requestFileHandles([makeHashDescriptor(hashHex)], {
            create: true,
        });
        const writableStream = await handle.createWritable();
        await writableStream.write(blob);
        await writableStream.close();
    };

    /**
     * Background task for non-LFS resources: consumes `stream`, computes the SHA-256 hash
     * of the resulting blob, stores it in cross-origin storage, and persists the computed
     * hash to `HASH_CACHE_NAME` keyed by `request` so future `match` calls can resolve the
     * file without a network round-trip.
     *
     * Called fire-and-forget from `put` — errors are swallowed so failures never surface to
     * the caller.
     *
     * @param {string} request The original resource URL.
     * @param {ReadableStream} stream The response body stream to consume.
     * @returns {Promise<void>}
     */
    _processAndStore = async (request, stream) => {
        try {
            const chunks = [];
            for await (const chunk of stream) {
                chunks.push(chunk);
            }
            const blob = new Blob(chunks);
            const hashHex = await this._getBlobHash(blob);

            await this._storeBlobInCOS(blob, hashHex);

            // Persist the computed hash so future match() calls resolve without the network.
            try {
                const hashCache = await this._getHashCache();
                await hashCache.put(request, new Response(hashHex));
            } catch {
                // Cache API unavailable (e.g. non-secure context): COS entry still written.
            }
        } catch {
            // Non-fatal: background store failure must not affect the caller.
        }
    };

    /**
     * Deletes the cache entry for the given request.
     *
     * Removes the hash entry from `HASH_CACHE_NAME`. Note: cross-origin storage itself does not
     * expose a delete API, so only the local hash mapping is removed. For non-LFS URLs this
     * permanently prevents `match` from resolving the file. For LFS-backed URLs, `match` will
     * re-fetch the LFS pointer file on the next call and repopulate the hash cache automatically.
     *
     * Implements `CacheInterface.delete`.
     *
     * @param {string} request
     * @returns {Promise<boolean>} Resolves to `true` if the hash entry was deleted, `false` otherwise.
     */
    delete = async (request) => {
        try {
            const hashCache = await this._getHashCache();
            return await hashCache.delete(request);
        } catch {
            return false;
        }
    };

    /**
     * Resolves the SHA-256 hash for a given URL.
     *
     * Returns the cached hash immediately if one has been persisted to `HASH_CACHE_NAME`.
     * Otherwise falls back to `_getLfsFileHash` to retrieve the hash from the Hugging Face
     * LFS pointer file, persisting the result to `HASH_CACHE_NAME` for future lookups.
     *
     * Returns `null` if the hash cannot be determined (e.g. non-LFS URL with no cached entry).
     *
     * @param {string} url The resource URL to resolve a hash for.
     * @returns {Promise<string|null>} The hex-encoded SHA-256 hash, or `null` if unavailable.
     */
    _getFileHash = async (url) => {
        try {
            const hashCache = await this._getHashCache();
            const cached = await hashCache.match(url);
            if (cached) {
                return cached.text();
            }

            const hash = await this._getLfsFileHash(url);
            if (hash) {
                await hashCache.put(url, new Response(hash));
                return hash;
            }

            return null;
        } catch {
            return null;
        }
    };

    /**
     * Attempts to retrieve the SHA-256 hash for a Hugging Face resource URL from its raw
     * Git LFS pointer file.
     *
     * Only applicable to URLs containing `/resolve/` (i.e. Hugging Face resolved file URLs).
     * The `/resolve/` segment is rewritten to `/raw/` to fetch the LFS pointer directly.
     * Returns `null` for non-LFS URLs or when the network request fails.
     *
     * @see https://huggingface.co/docs/hub/en/storage-backends#xet
     * @param {string} url The resolved Hugging Face URL of the resource.
     * @returns {Promise<string|null>} The hex-encoded SHA-256 hash, or `null` if unavailable.
     */
    _getLfsFileHash = async (url) => {
        if (!url.includes('/resolve/')) {
            return null;
        }

        const rawUrl = url.replace('/resolve/', '/raw/');

        try {
            const text = await fetch(rawUrl).then((r) => r.text());
            const match = text.match(/^oid sha256:([0-9a-f]+)$/m);
            return match ? match[1] : null;
        } catch {
            return null;
        }
    };

    /**
     * Computes the SHA-256 hash of a `Blob`'s contents.
     *
     * @param {Blob} blob The blob to hash.
     * @returns {Promise<string>} The lowercase hex-encoded SHA-256 hash.
     */
    _getBlobHash = async (blob) => {
        const arrayBuffer = await blob.arrayBuffer();
        const hashBuffer = await crypto.subtle.digest(HASH_ALGORITHM, arrayBuffer);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        return hashArray.map((byte) => byte.toString(16).padStart(2, '0')).join('');
    };
}
