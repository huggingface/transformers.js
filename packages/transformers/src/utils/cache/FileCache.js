import fs from 'node:fs';
import path from 'node:path';

import { FileResponse } from '../hub/FileResponse.js';
import { Random } from '../random.js';
import { apis } from '../../env.js';

// Create a dedicated random instance for generating unique temporary file names
const rng = new Random();

// How long a `.incomplete.lock` may sit before another writer treats it as
// stale (a previous run crashed without releasing it). Well above any realistic
// single-chunk write, but low enough that a resume is not blocked forever.
const LOCK_STALE_MS = 10 * 60 * 1000;

/**
 * File system cache implementation that implements the CacheInterface.
 * Provides `match` and `put` methods compatible with the Web Cache API.
 *
 * Large downloads are streamed to a deterministic `<key>.incomplete` file with
 * a small sidecar recording the server `ETag` and expected size. If a download
 * is interrupted the partial is kept, so a later attempt can send a `Range`
 * request (see `getResumeInfo`) and continue instead of starting from byte 0.
 */
export class FileCache {
    /**
     * Instantiate a `FileCache` object.
     * @param {string} path
     */
    constructor(path) {
        this.path = path;
    }

    /**
     * Checks whether the given request is in the cache.
     * @param {string} request
     * @returns {Promise<FileResponse | undefined>}
     */
    async match(request) {
        let filePath = path.join(this.path, request);
        let file = new FileResponse(filePath);

        if (file.exists) {
            return file;
        } else {
            return undefined;
        }
    }

    /**
     * Report whether a resumable partial download exists for `request`.
     * Returns `{ size, etag, total }` when a `.incomplete` file and its sidecar
     * are present and internally consistent, otherwise `undefined`. Callers use
     * this to send a `Range`/`If-Range` request and continue the download.
     * @param {string} request
     * @returns {Promise<{size: number, etag: string|null, total: number} | undefined>}
     */
    async getResumeInfo(request) {
        const incompletePath = path.join(this.path, request) + '.incomplete';
        const sidecarPath = incompletePath + '.json';
        try {
            const [stat, sidecarRaw] = await Promise.all([
                fs.promises.stat(incompletePath),
                fs.promises.readFile(sidecarPath, 'utf-8'),
            ]);
            const sidecar = JSON.parse(sidecarRaw);
            const total = typeof sidecar.total === 'number' ? sidecar.total : 0;
            // Only resume when there is something to resume and we have not
            // somehow written past the expected size. Anything inconsistent
            // falls through to a clean restart.
            if (stat.size > 0 && (total === 0 || stat.size < total)) {
                return { size: stat.size, etag: sidecar.etag ?? null, total };
            }
        } catch {
            // No partial, no sidecar, or unreadable — nothing to resume.
        }
        return undefined;
    }

    /**
     * Adds the given response to the cache.
     * @param {string} request
     * @param {Response} response
     * @param {(data: {progress: number, loaded: number, total: number}) => void} [progress_callback] Optional.
     * The function to call with progress updates
     * @returns {Promise<void>}
     */
    async put(request, response, progress_callback = undefined) {
        const filePath = path.join(this.path, request);
        await fs.promises.mkdir(path.dirname(filePath), { recursive: true });

        const incompletePath = filePath + '.incomplete';
        const sidecarPath = incompletePath + '.json';
        const lockPath = incompletePath + '.lock';

        // Claim the deterministic partial for this key. If another writer (this
        // or another process) already holds it, fall back to the legacy
        // random-suffix temp path: correct and non-resumable, but it never
        // corrupts the shared partial — preserving the previous concurrency
        // guarantee for parallel loads of the same file.
        const locked = await this.#acquireLock(lockPath);
        if (!locked) {
            return this.#putUniqueTemp(filePath, response, progress_callback);
        }

        // A 206 (Partial Content) means the server honored our Range request and
        // we append to the existing partial. Anything else (200) is a fresh
        // download: discard any stale partial and start over.
        const resuming = response.status === 206;
        const total = this.#expectedTotal(response, resuming);

        let loaded = 0;
        if (resuming) {
            try {
                loaded = (await fs.promises.stat(incompletePath)).size;
            } catch {
                loaded = 0;
            }
        }

        try {
            // Record the etag + expected total up front so an interrupted write
            // still leaves a sidecar the next attempt can validate against.
            await fs.promises.writeFile(sidecarPath, JSON.stringify({ etag: response.headers.get('etag'), total }));

            const fileStream = fs.createWriteStream(incompletePath, {
                flags: resuming ? 'a' : 'w',
            });
            const reader = response.body.getReader();

            while (true) {
                const { done, value } = await reader.read();
                if (done) {
                    break;
                }

                await new Promise((resolve, reject) => {
                    fileStream.write(value, (err) => (err ? reject(err) : resolve()));
                });

                loaded += value.length;
                const progress = total ? (loaded / total) * 100 : 0;

                progress_callback?.({ progress, loaded, total });
            }

            await new Promise((resolve, reject) => {
                fileStream.close((err) => (err ? reject(err) : resolve()));
            });

            // Guard against a truncated body: if we know the expected size and
            // fell short, keep the partial so the next call can resume rather
            // than promoting an incomplete file to the final path.
            if (total && loaded < total) {
                throw new Error(`Incomplete download for "${request}": got ${loaded} of ${total} bytes.`);
            }

            // Atomically publish the completed file and drop the sidecar.
            await fs.promises.rename(incompletePath, filePath);
            await fs.promises.unlink(sidecarPath).catch(() => {});
        } catch (error) {
            // Intentionally keep `.incomplete` + sidecar on failure so a later
            // attempt can resume from here. Only the lock is released (finally).
            throw error;
        } finally {
            await fs.promises.unlink(lockPath).catch(() => {});
        }
    }

    /**
     * Legacy write path: stream to a unique temp file and atomically rename.
     * Used when the deterministic partial is already locked by a concurrent
     * writer, so this call must not touch the shared `.incomplete` file.
     * @param {string} filePath
     * @param {Response} response
     * @param {((data: {progress: number, loaded: number, total: number}) => void) | undefined} progress_callback
     * @returns {Promise<void>}
     */
    async #putUniqueTemp(filePath, response, progress_callback) {
        const id = apis.IS_PROCESS_AVAILABLE ? process.pid : Date.now();
        const randomSuffix = rng._int32().toString(36);
        const tmpPath = filePath + `.tmp.${id}.${randomSuffix}`;

        try {
            const total = parseInt(response.headers.get('Content-Length') ?? '0');
            let loaded = 0;

            const fileStream = fs.createWriteStream(tmpPath);
            const reader = response.body.getReader();

            while (true) {
                const { done, value } = await reader.read();
                if (done) {
                    break;
                }

                await new Promise((resolve, reject) => {
                    fileStream.write(value, (err) => (err ? reject(err) : resolve()));
                });

                loaded += value.length;
                const progress = total ? (loaded / total) * 100 : 0;

                progress_callback?.({ progress, loaded, total });
            }

            await new Promise((resolve, reject) => {
                fileStream.close((err) => (err ? reject(err) : resolve()));
            });

            await fs.promises.rename(tmpPath, filePath);
        } catch (error) {
            try {
                await fs.promises.unlink(tmpPath);
            } catch {}
            throw error;
        }
    }

    /**
     * Total expected file size. For a 206 response it is read from the
     * `Content-Range: bytes start-end/total` header; for a fresh 200 from
     * `Content-Length`. Returns 0 when the size is unknown.
     * @param {Response} response
     * @param {boolean} resuming
     * @returns {number}
     */
    #expectedTotal(response, resuming) {
        if (resuming) {
            const contentRange = response.headers.get('Content-Range');
            const match = contentRange && /\/(\d+)\s*$/.exec(contentRange);
            if (match) {
                return parseInt(match[1], 10);
            }
        }
        return parseInt(response.headers.get('Content-Length') ?? '0', 10) || 0;
    }

    /**
     * Acquire the per-key write lock via an exclusive create (`wx`). If a lock
     * already exists but is older than `LOCK_STALE_MS`, it is treated as
     * abandoned by a crashed writer and stolen.
     * @param {string} lockPath
     * @returns {Promise<boolean>}
     */
    async #acquireLock(lockPath) {
        try {
            const fd = await fs.promises.open(lockPath, 'wx');
            await fd.close();
            return true;
        } catch (e) {
            if (e.code !== 'EEXIST') {
                return false;
            }
            // Steal a stale lock left behind by a crashed process.
            try {
                const stat = await fs.promises.stat(lockPath);
                if (Date.now() - stat.mtimeMs > LOCK_STALE_MS) {
                    await fs.promises.unlink(lockPath).catch(() => {});
                    const fd = await fs.promises.open(lockPath, 'wx');
                    await fd.close();
                    return true;
                }
            } catch {}
            return false;
        }
    }

    /**
     * Deletes the cache entry for the given request.
     * @param {string} request
     * @returns {Promise<boolean>} A Promise that resolves to `true` if the cache entry was deleted, `false` otherwise.
     */
    async delete(request) {
        let filePath = path.join(this.path, request);

        try {
            await fs.promises.unlink(filePath);
            return true;
        } catch (error) {
            // File doesn't exist or couldn't be deleted
            return false;
        }
    }

    // TODO add the rest?
    // addAll(requests: RequestInfo[]): Promise<void>;
    // keys(request?: RequestInfo | URL, options?: CacheQueryOptions): Promise<ReadonlyArray<Request>>;
    // match(request: RequestInfo | URL, options?: CacheQueryOptions): Promise<Response | undefined>;
    // matchAll(request?: RequestInfo | URL, options?: CacheQueryOptions): Promise<ReadonlyArray<Response>>;
}
