import fs from 'node:fs';
import path from 'node:path';
import { Readable } from 'node:stream';
import { pipeline as pipe } from 'node:stream/promises';
import { setTimeout as delay } from 'node:timers/promises';

import { FileResponse } from '../hub/FileResponse.js';
import { Random } from '../random.js';
import { apis } from '../../env.js';

// Create a dedicated random instance for generating unique temporary file names
const rng = new Random();

// Only publication and deletion hold this lock; downloads are staged outside it. Never steal a
// lock: a paused owner may still be alive. A crash leaves an actionable, bounded failure instead
// of risking another writer's data. Remove an abandoned lock only after confirming no owner is active.
// Cross-process mutation safety: https://github.com/huggingface/transformers.js/issues/1933
async function withMutationLock(markerPath, operation) {
    const lockPath = `${markerPath}.lock`;
    const deadline = performance.now() + 1000;
    let handle;
    while (handle === undefined) {
        try {
            handle = await fs.promises.open(lockPath, 'wx');
        } catch (error) {
            if (error.code !== 'EEXIST') throw error;
            if (performance.now() >= deadline) {
                throw new Error(
                    `Timed out waiting for cache mutation lock: ${lockPath}. Check for an active or crashed writer.`,
                    { cause: error },
                );
            }
            await delay(10);
        }
    }
    try {
        return await operation();
    } finally {
        try {
            await handle.close();
        } finally {
            await fs.promises.unlink(lockPath);
        }
    }
}

/**
 * File system cache implementation that implements the CacheInterface.
 * Provides `match` and `put` methods compatible with the Web Cache API.
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
     * Translate cache identity to storage layout. Versioned keys are stored at the
     * same path as unversioned ones, so the cache keeps the layout of the repo; the
     * version lives in a marker file next to the cached file (no marker = version 0).
     *
     * Releases before versioning don't know about markers. If one shares this folder
     * and deletes and re-downloads a versioned file, the old marker stays and the
     * re-downloaded file counts as current. That case is rare, so it is accepted.
     * @private
     * @param {string} request
     * @returns {{ filePath: string, markerPath: string, version: number }}
     */
    resolve(request) {
        let version = 0;
        const key = request.replace(/([?&])__transformersjs_cache_version=(\d+)(&?)/, (_, separator, value, next) => {
            version = Number(value);
            return next ? separator : '';
        });
        const filePath = path.join(this.path, key);
        const markerPath = path.join(
            path.dirname(filePath),
            `.${path.basename(filePath)}.transformersjs_cache_version`,
        );
        return { filePath, markerPath, version };
    }

    /**
     * Whether two logical keys alias the same physical entry (FileCache-only cleanup detail).
     * @param {string} request
     * @param {string} other
     * @returns {boolean}
     * @internal
     */
    isSameFile(request, other) {
        return this.resolve(request).filePath === this.resolve(other).filePath;
    }

    /**
     * @private
     * @param {string} markerPath
     * @returns {Promise<number>}
     */
    async readVersion(markerPath) {
        try {
            return Number(await fs.promises.readFile(markerPath, 'utf8')) || 0;
        } catch {
            return 0;
        }
    }

    /**
     * Checks whether the given request is in the cache.
     * @param {string} request
     * @returns {Promise<FileResponse | undefined>}
     */
    async match(request) {
        const { filePath, markerPath, version } = this.resolve(request);
        let file = new FileResponse(filePath);

        if (file.exists && (await this.readVersion(markerPath)) === version) {
            return file;
        } else {
            return undefined;
        }
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
        const { filePath, markerPath, version } = this.resolve(request);

        // Include both PID and a random suffix so that concurrent put() call within the same process
        // (e.g., multiple pipelines loading the same file in parallel) each get their own temp file
        // and don't corrupt each other's writes.
        const id = apis.IS_PROCESS_AVAILABLE ? process.pid : Date.now();
        const randomSuffix = rng._int32().toString(36);
        const tmpPath = filePath + `.tmp.${id}.${randomSuffix}`;
        const markerTmpPath = markerPath + `.tmp.${id}.${randomSuffix}`;

        try {
            const contentLength = response.headers.get('Content-Length');
            const total = parseInt(contentLength ?? '0');
            let loaded = 0;

            await fs.promises.mkdir(path.dirname(filePath), { recursive: true });
            // As in saveBlob, pipeline handles stream errors and waits for close before cleanup or rename.
            await pipe(
                Readable.fromWeb(response.body),
                async function* (source) {
                    for await (const value of source) {
                        yield value;
                        loaded += value.length;
                        const progress = total ? (loaded / total) * 100 : 0;
                        progress_callback?.({ progress, loaded, total });
                    }
                },
                fs.createWriteStream(tmpPath),
            );

            if (version) await fs.promises.writeFile(markerTmpPath, String(version), { flag: 'wx' });
            // Every cooperating process serializes payload/marker publication with delete's
            // version check and both unlinks. A stale delete either finishes before publication,
            // or observes the new version and refuses deletion. Publish the complete marker last.
            await withMutationLock(markerPath, async () => {
                await fs.promises.rename(tmpPath, filePath);
                if (version) await fs.promises.rename(markerTmpPath, markerPath);
                else await fs.promises.rm(markerPath, { force: true });
            });
        } catch (error) {
            // Clean up the temp file if an error occurred during download
            try {
                await fs.promises.unlink(tmpPath);
            } catch {}
            try {
                await fs.promises.unlink(markerTmpPath);
            } catch {}
            throw error;
        }
    }

    /**
     * Deletes the cache entry for the given request.
     * @param {string} request
     * @returns {Promise<boolean>} A Promise that resolves to `true` if the cache entry was deleted, `false` otherwise.
     */
    async delete(request) {
        const { filePath, markerPath, version } = this.resolve(request);

        try {
            if (!fs.existsSync(filePath)) return false;
            return await withMutationLock(markerPath, async () => {
                if ((await this.readVersion(markerPath)) !== version) return false;
                await fs.promises.unlink(filePath);
                await fs.promises.rm(markerPath, { force: true });
                return true;
            });
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
