/**
 * @file Origin Private File System (OPFS) cache backend.
 *
 * Provides persistent browser storage via the Origin Private File System.
 *
 * @module utils/cache/OPFSCache
 */

/**
 * @param {unknown} err
 * @returns {boolean}
 */
const isNotFound = (err) => err instanceof DOMException && err.name === 'NotFoundError';

/**
 * Resolves a request URL into OPFS directory segments and a filename.
 * @param {string} request
 * @returns {{ dir: string[], file: string }}
 */
function resolvePath(request) {
    const pathname = URL.canParse(request) ? new URL(request).pathname : request;

    const segments = pathname.split('/').filter((s) => s.length > 0);

    if (segments.length === 0) {
        throw new TypeError('Path resolved to zero segments');
    }

    const file = segments.pop();
    return { dir: segments, file };
}

/**
 * OPFS-backed cache implementation.
 *
 * Implements {@link import('../cache.js').CacheInterface}.
 */
export class OPFSCache {
    /** @type {Promise<FileSystemDirectoryHandle> | null} */
    #rootPromise = null;

    /** @type {string} */
    #rootName;

    /**
     * @param {string} rootName The OPFS subdirectory name used as the cache root.
     */
    constructor(rootName) {
        this.#rootName = rootName;
    }

    /**
     * Returns whether the OPFS API is available in the current environment.
     * @returns {boolean}
     */
    static isAvailable = () =>
        typeof navigator !== 'undefined' && typeof navigator.storage?.getDirectory === 'function';

    /**
     * @returns {Promise<FileSystemDirectoryHandle>}
     */
    _getRoot() {
        this.#rootPromise ??= navigator.storage
            .getDirectory()
            .then((opfs) => opfs.getDirectoryHandle(this.#rootName, { create: true }));
        return this.#rootPromise;
    }

    /**
     * Walk a chain of nested directories.
     * @param {string[]} segments
     * @param {boolean} create
     * @returns {Promise<FileSystemDirectoryHandle | undefined>}
     */
    async _navigate(segments, create) {
        let dir = await this._getRoot();
        for (const seg of segments) {
            try {
                dir = await dir.getDirectoryHandle(seg, { create });
            } catch (err) {
                if (!create && isNotFound(err)) return undefined;
                throw err;
            }
        }
        return dir;
    }

    /**
     * Looks up a cached response for the given URL.
     *
     * Implements `CacheInterface.match`.
     *
     * @param {string} request The URL of the resource to look up.
     * @returns {Promise<Response | undefined>} The cached `Response`, or `undefined` if not found.
     */
    match = async (request) => {
        const { dir: dirSegments, file: fileName } = resolvePath(request);

        const dir = await this._navigate(dirSegments, false);
        if (dir === undefined) return undefined;

        try {
            const file = await dir.getFileHandle(fileName).then((h) => h.getFile());

            return new Response(file.stream(), {
                headers: {
                    'Content-Length': String(file.size),
                },
            });
        } catch (err) {
            if (isNotFound(err)) return undefined;
            throw err;
        }
    };

    /**
     * Stores a response in the OPFS cache.
     *
     * Implements `CacheInterface.put`.
     *
     * @param {string} request The URL of the resource.
     * @param {Response} response The response whose body will be written to the cache.
     * @returns {Promise<void>}
     */
    put = async (request, response) => {
        const { dir: dirSegments, file: fileName } = resolvePath(request);

        const dir = await this._navigate(dirSegments, true);

        const fileHandle = await dir.getFileHandle(fileName, { create: true });
        const writable = await fileHandle.createWritable();
        if (response.body === null) {
            await writable.close();
        } else {
            await response.body.pipeTo(writable);
        }
    };

    /**
     * Deletes the cache entry for the given request.
     *
     * Implements `CacheInterface.delete`.
     *
     * @param {string} request
     * @returns {Promise<boolean>}
     */
    delete = async (request) => {
        const { dir: dirSegments, file: fileName } = resolvePath(request);

        const dir = await this._navigate(dirSegments, false);
        if (dir === undefined) return false;

        try {
            await dir.removeEntry(fileName);
            return true;
        } catch (err) {
            if (isNotFound(err)) return false;
            throw err;
        }
    };
}
