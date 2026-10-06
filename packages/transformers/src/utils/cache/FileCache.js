import fs from 'node:fs';
import path from 'node:path';

import { FileResponse } from '../hub/FileResponse.js';
import { Random } from '../random.js';
import { apis } from '../../env.js';

// Create a dedicated random instance for generating unique temporary file names
const rng = new Random();

// Distinguishes this process from any other lock holder. A pid alone is not
// enough: on a shared/network cache directory the same pid can exist on a
// different machine, and we must not read that as "our own writer is alive".
const INSTANCE_ID = rng._int32().toString(36);

// A live writer refreshes its lock on this interval, so a lock that has not
// been touched for `LOCK_STALE_MS` belongs to a writer that is gone — not to a
// slow download. The heartbeat is what makes the staleness check safe.
const LOCK_HEARTBEAT_MS = 30 * 1000;
const LOCK_STALE_MS = 5 * 60 * 1000;

// Absolute ceiling. Reached only when the lock looks stale but its owner claims
// a pid that still resolves, which on a shared cache directory may be an
// unrelated process on another host. Prevents a permanent deadlock.
const LOCK_ABANDONED_MS = 60 * 60 * 1000;

/**
 * Destroy a write stream and wait until its file descriptor is actually
 * released. Used on the error paths, where the stream is still open over a
 * partial or temp file that is about to be unlinked or handed to another
 * writer.
 *
 * The early-out tests `closed`, not `destroyed`: `destroyed` is set
 * synchronously when teardown *begins*, so a stream already destroyed by an
 * auto-destroy on error can still be holding its descriptor. `closed` flips
 * with the `close` event, which is emitted once the descriptor is gone.
 *
 * @param {import('node:fs').WriteStream | undefined} stream
 * @returns {Promise<void>}
 */
function destroyStream(stream) {
    if (!stream || stream.closed) {
        return Promise.resolve();
    }
    return new Promise((resolve) => {
        // `close` fires whether the teardown itself succeeded or not — either
        // way there is nothing further to wait for.
        stream.once('close', resolve);
        // An already-destroyed stream is mid-teardown; destroying it again is
        // pointless, but its pending `close` is still worth waiting for.
        if (!stream.destroyed) {
            stream.destroy();
        }
    });
}

/**
 * File system cache implementation that implements the CacheInterface.
 * Provides `match` and `put` methods compatible with the Web Cache API.
 *
 * Large downloads are streamed to a deterministic `<key>.incomplete` file with
 * a small sidecar recording the server `ETag` and expected size. If a download
 * is interrupted the partial is kept, so a later attempt can send a `Range`
 * request and continue instead of starting from byte 0.
 *
 * Resuming is coordinated by a per-key lock that is taken *before* the request
 * is issued (see `reserveResume`). A caller may only send `Range` for a key it
 * has reserved, which means a partial response can never reach a writer that
 * does not own the corresponding partial file.
 */
export class FileCache {
    /**
     * Locks held by this instance, keyed by request. Populated by
     * `reserveResume` and consumed by `put`/`releaseResume`.
     * @type {Map<string, {lockPath: string, timer: any}>}
     */
    #reservations = new Map();

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
     * Reserve a resumable partial download for `request`.
     *
     * Takes the per-key write lock and, only if a consistent partial exists,
     * reports `{ size, etag, total }` so the caller can continue the download
     * with a `Range`/`If-Range` request. Returns `undefined` when there is
     * nothing to resume or another writer holds the key, in which case the
     * caller must perform an ordinary full request.
     *
     * Taking the lock here (rather than in `put`) is what guarantees a `206`
     * is only ever produced by the writer that owns the partial it continues.
     * A successful reservation is released by the matching `put`, or by
     * `releaseResume` if the caller abandons the download.
     *
     * @param {string} request
     * @returns {Promise<import('../cache.js').ResumeInfo | undefined>}
     */
    async reserveResume(request) {
        const incompletePath = path.join(this.path, request) + '.incomplete';
        const sidecarPath = incompletePath + '.json';

        let info;
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
            if (stat.size > 0 && total > 0 && stat.size < total) {
                info = { size: stat.size, etag: sidecar.etag ?? null, total };
            }
        } catch {
            // No partial, no sidecar, or unreadable — nothing to resume.
        }
        if (!info) {
            return undefined;
        }

        // Only claim the lock once we know a resume is actually possible, so a
        // pointless reservation never blocks another writer.
        if (!(await this.#acquireLock(request))) {
            return undefined;
        }

        // Re-check under the lock: another writer may have completed or reset
        // the partial between the stat above and the lock being granted.
        try {
            const stat = await fs.promises.stat(incompletePath);
            if (stat.size !== info.size) {
                await this.releaseResume(request);
                return undefined;
            }
        } catch {
            await this.releaseResume(request);
            return undefined;
        }

        return info;
    }

    /**
     * Release a reservation taken by `reserveResume` without writing anything.
     * Safe to call when no reservation is held.
     * @param {string} request
     * @returns {Promise<void>}
     */
    async releaseResume(request) {
        const reservation = this.#reservations.get(request);
        if (!reservation) {
            return;
        }
        this.#reservations.delete(request);
        clearInterval(reservation.timer);
        await fs.promises.unlink(reservation.lockPath).catch(() => {});
    }

    /**
     * Adds the given response to the cache.
     *
     * A `206 Partial Content` response is only accepted when this instance
     * holds a reservation for `request` (see `reserveResume`) and the
     * `Content-Range` header lines up exactly with the partial on disk;
     * otherwise the write is refused rather than risking a corrupt entry.
     *
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
        const resuming = response.status === 206;

        // A partial response is only meaningful to the writer that owns the
        // partial it continues. Without a reservation we have no such partial,
        // so writing this body anywhere would publish a fragment as a whole
        // file. Refuse instead — the caller must retry without `Range`.
        const reserved = this.#reservations.has(request);
        if (resuming && !reserved) {
            throw new Error(
                `Refusing to cache a partial (206) response for "${request}" without a resume reservation. ` +
                    `Retry the request without a \`Range\` header.`,
            );
        }

        // Not resuming and no reservation: fall back to the legacy
        // random-suffix temp path. Correct and non-resumable, but it never
        // touches the shared partial — preserving the previous concurrency
        // guarantee for parallel loads of the same file.
        if (!reserved && !(await this.#acquireLock(request))) {
            return this.#putUniqueTemp(filePath, response, progress_callback);
        }

        /** @type {import('node:fs').WriteStream | undefined} */
        let fileStream;
        try {
            let loaded = 0;
            let total;

            if (resuming) {
                const partialSize = await fs.promises.stat(incompletePath).then(
                    (s) => s.size,
                    () => 0,
                );
                const range = await this.#validateContentRange(response, request, partialSize, sidecarPath);
                loaded = range.start;
                total = range.total;
            } else {
                // A fresh download supersedes any partial we were holding.
                total = parseInt(response.headers.get('Content-Length') ?? '0', 10) || 0;
            }

            // Record the etag + expected total up front so an interrupted write
            // still leaves a sidecar the next attempt can validate against. On a
            // 206 the validator is the one we resumed against, not the (absent)
            // ETag of the partial response.
            if (!resuming) {
                await fs.promises.writeFile(sidecarPath, JSON.stringify({ etag: response.headers.get('etag'), total }));
            }

            fileStream = fs.createWriteStream(incompletePath, {
                flags: resuming ? 'a' : 'w',
            });
            // Write failures are surfaced through the `write`/`close` callbacks
            // below. Without a listener the same failure also reaches the
            // process as an unhandled `error` event.
            fileStream.on('error', () => {});
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

            const stream = fileStream;
            await new Promise((resolve, reject) => {
                stream.close((err) => (err ? reject(err) : resolve()));
            });
            // Closed cleanly, so `finally` has nothing left to tear down.
            fileStream = undefined;

            // The body must match the size we were promised, exactly. Falling
            // short keeps the partial so the next call can resume; an oversized
            // body is a broken server or a changed file, and cannot be resumed
            // onto (the partial now runs past `total`), so it is discarded and
            // the next attempt starts clean. Neither case is ever published.
            if (total && loaded !== total) {
                if (loaded > total) {
                    await fs.promises.unlink(incompletePath).catch(() => {});
                    await fs.promises.unlink(sidecarPath).catch(() => {});
                    throw new Error(
                        `Oversized download for "${request}": got ${loaded} bytes but expected ${total}. ` +
                            `Discarded the partial; retry from scratch.`,
                    );
                }
                throw new Error(`Incomplete download for "${request}": got ${loaded} of ${total} bytes.`);
            }

            // Atomically publish the completed file and drop the sidecar.
            await fs.promises.rename(incompletePath, filePath);
            await fs.promises.unlink(sidecarPath).catch(() => {});
        } finally {
            // A reader, writer, or progress-callback error leaves the stream
            // open on the partial. Tear it down *before* the reservation is
            // released, so the next writer — which may be another process
            // taking this key the moment the lock disappears — never finds a
            // file we are still holding open.
            await destroyStream(fileStream);

            // `.incomplete` + sidecar are intentionally kept on failure so a
            // later attempt can resume from here. Only the lock is released.
            await this.releaseResume(request);
        }
    }

    /**
     * Validate a `206` response against the partial on disk.
     *
     * Every component of `Content-Range: bytes <start>-<end>/<total>` is
     * checked: the range must begin exactly where the partial ends (no gap and
     * no overlap), its length must agree with `Content-Length`, and its total
     * must match what the sidecar recorded. Unsatisfied (`bytes *`/`<total>`)
     * and suffix (`bytes -500/1234`) forms are rejected outright — neither can
     * be appended safely.
     *
     * A mismatch means our assumptions about the partial no longer hold, so it
     * is discarded and the caller restarts from byte 0 rather than resuming
     * onto data that may belong to a different revision of the file.
     *
     * @param {Response} response
     * @param {string} request
     * @param {number} partialSize
     * @param {string} sidecarPath
     * @returns {Promise<{start: number, end: number, total: number}>}
     */
    async #validateContentRange(response, request, partialSize, sidecarPath) {
        const raw = response.headers.get('Content-Range');
        const match = /^\s*bytes\s+(\d+)-(\d+)\/(\d+)\s*$/.exec(raw ?? '');

        const reject = async (reason) => {
            const incompletePath = sidecarPath.replace(/\.json$/, '');
            await fs.promises.unlink(incompletePath).catch(() => {});
            await fs.promises.unlink(sidecarPath).catch(() => {});
            return new Error(`Cannot resume "${request}": ${reason}. Discarded the partial; retry from scratch.`);
        };

        if (!match) {
            throw await reject(`unusable Content-Range "${raw ?? '<missing>'}"`);
        }

        const start = parseInt(match[1], 10);
        const end = parseInt(match[2], 10);
        const total = parseInt(match[3], 10);

        if (end < start || end >= total) {
            throw await reject(`inconsistent Content-Range "${raw}"`);
        }
        if (start !== partialSize) {
            throw await reject(
                `server resumed at byte ${start} but the partial holds ${partialSize} bytes ` +
                    `(${start > partialSize ? 'gap' : 'overlap'})`,
            );
        }

        const contentLength = response.headers.get('Content-Length');
        if (contentLength !== null) {
            const expected = end - start + 1;
            const declared = parseInt(contentLength, 10);
            if (declared !== expected) {
                throw await reject(`Content-Length ${declared} does not match Content-Range span ${expected}`);
            }
        }

        try {
            const sidecar = JSON.parse(await fs.promises.readFile(sidecarPath, 'utf-8'));
            if (typeof sidecar.total === 'number' && sidecar.total > 0 && sidecar.total !== total) {
                throw await reject(`total size changed from ${sidecar.total} to ${total}`);
            }
        } catch (e) {
            if (e instanceof Error && e.message.startsWith('Cannot resume')) {
                throw e;
            }
            // An unreadable sidecar is not fatal: the Content-Range checks above
            // already pin the append to the right offset.
        }

        return { start, end, total };
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
        // This path renames whatever it writes onto the final file, so it must
        // only ever see a complete body. A partial response reaching here would
        // publish a fragment as the whole file.
        if (response.status !== 200) {
            throw new Error(
                `Refusing to cache a ${response.status} response as a complete file for "${filePath}". ` +
                    `Only a full 200 response can be written by the fallback path.`,
            );
        }

        const id = apis.IS_PROCESS_AVAILABLE ? process.pid : Date.now();
        const randomSuffix = rng._int32().toString(36);
        const tmpPath = filePath + `.tmp.${id}.${randomSuffix}`;

        /** @type {import('node:fs').WriteStream | undefined} */
        let fileStream;
        try {
            const total = parseInt(response.headers.get('Content-Length') ?? '0');
            let loaded = 0;

            fileStream = fs.createWriteStream(tmpPath);
            fileStream.on('error', () => {});
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

            const stream = fileStream;
            await new Promise((resolve, reject) => {
                stream.close((err) => (err ? reject(err) : resolve()));
            });
            fileStream = undefined;

            // This path renames onto the final file, so the body must match the
            // declared size exactly — an oversized one is no safer to publish
            // than a truncated one.
            if (total && loaded !== total) {
                throw new Error(
                    loaded > total
                        ? `Oversized download for "${filePath}": got ${loaded} bytes but expected ${total}.`
                        : `Incomplete download for "${filePath}": got ${loaded} of ${total} bytes.`,
                );
            }

            await fs.promises.rename(tmpPath, filePath);
        } catch (error) {
            // Close the stream before unlinking: an open handle makes the
            // removal fail outright on Windows and leaves the temp file behind.
            await destroyStream(fileStream);
            try {
                await fs.promises.unlink(tmpPath);
            } catch {}
            throw error;
        }
    }

    /**
     * Acquire the per-key write lock via an exclusive create (`wx`), recording
     * the owner so its liveness can be checked later. While held, the lock is
     * refreshed on an interval — an untouched lock therefore means an absent
     * writer, never a slow one.
     *
     * An existing lock is only taken over when it has gone stale *and* its
     * owner is demonstrably gone, or when it has sat untouched long enough that
     * it must be treated as abandoned regardless.
     *
     * @param {string} request
     * @returns {Promise<boolean>}
     */
    async #acquireLock(request) {
        const lockPath = path.join(this.path, request) + '.incomplete.lock';

        if (!(await this.#writeLockFile(lockPath))) {
            if (!(await this.#isLockAbandoned(lockPath))) {
                return false;
            }
            await fs.promises.unlink(lockPath).catch(() => {});
            if (!(await this.#writeLockFile(lockPath))) {
                // Another writer won the race for the abandoned lock.
                return false;
            }
        }

        // Keep the lock warm for as long as we hold it, so other writers can
        // tell an in-progress download from a crashed one.
        const timer = setInterval(() => {
            const now = new Date();
            fs.promises.utimes(lockPath, now, now).catch(() => {});
        }, LOCK_HEARTBEAT_MS);
        timer.unref?.();

        this.#reservations.set(request, { lockPath, timer });
        return true;
    }

    /**
     * Exclusively create `lockPath` and stamp it with this writer's identity.
     * @param {string} lockPath
     * @returns {Promise<boolean>} Whether the lock was created.
     */
    async #writeLockFile(lockPath) {
        try {
            const handle = await fs.promises.open(lockPath, 'wx');
            try {
                await handle.writeFile(
                    JSON.stringify({
                        instance: INSTANCE_ID,
                        pid: apis.IS_PROCESS_AVAILABLE ? process.pid : null,
                        startedAt: Date.now(),
                    }),
                );
            } finally {
                await handle.close();
            }
            return true;
        } catch {
            return false;
        }
    }

    /**
     * Whether an existing lock may be taken over. True only when the lock has
     * not been refreshed for `LOCK_STALE_MS` and its owning process is gone, or
     * when it has sat untouched for `LOCK_ABANDONED_MS` and must be reclaimed
     * to avoid deadlocking on an owner we cannot verify.
     * @param {string} lockPath
     * @returns {Promise<boolean>}
     */
    async #isLockAbandoned(lockPath) {
        let age;
        try {
            age = Date.now() - (await fs.promises.stat(lockPath)).mtimeMs;
        } catch {
            // The lock vanished — treat it as gone so the caller retries.
            return true;
        }

        if (age < LOCK_STALE_MS) {
            // Refreshed recently: a writer is actively downloading. Leave it be.
            return false;
        }
        if (age >= LOCK_ABANDONED_MS) {
            return true;
        }

        let owner;
        try {
            owner = JSON.parse(await fs.promises.readFile(lockPath, 'utf-8'));
        } catch {
            // Unreadable or truncated (e.g. a writer died mid-create): stale.
            return true;
        }

        // A lock stamped by this process but no longer tracked belongs to a
        // reservation we already released; reclaiming it is safe.
        if (owner?.instance === INSTANCE_ID) {
            return true;
        }
        if (!apis.IS_PROCESS_AVAILABLE || typeof owner?.pid !== 'number') {
            return true;
        }

        try {
            // Signal 0 performs the permission/existence check without sending.
            process.kill(owner.pid, 0);
            // The pid resolves, but on a shared cache directory it may belong to
            // an unrelated process on another host. Wait for the hard ceiling.
            return false;
        } catch (e) {
            // ESRCH: no such process. EPERM: alive but owned by another user.
            return e?.code === 'ESRCH';
        }
    }

    /**
     * Deletes the cache entry for the given request, including anything an
     * interrupted resumable download left behind: the `.incomplete` partial,
     * its sidecar, and the write lock.
     *
     * Removing only the completed file would leave a partial that a later
     * request resumes onto — reviving bytes the caller asked to delete — and a
     * lock sitting in the way of the next writer.
     *
     * An in-flight download by another writer is left alone: its lock, partial
     * and sidecar are that writer's working set, and removing them would let a
     * third writer take the same key and write concurrently. The completed
     * file is still removed in that case — it is the entry the caller asked to
     * delete, and it is not what the other writer is holding.
     *
     * @param {string} request
     * @returns {Promise<boolean>} A Promise that resolves to `true` if a cache
     * entry (the completed file or a partial download) was deleted, `false`
     * otherwise. A leftover sidecar or lock is cleaned up either way, but on
     * its own does not count as an entry.
     */
    async delete(request) {
        const filePath = path.join(this.path, request);
        const incompletePath = filePath + '.incomplete';
        const lockPath = incompletePath + '.lock';

        // Drop our own reservation first, so the lock is removed by the writer
        // that owns it and its heartbeat stops — rather than unlinking a path a
        // live timer would keep touching. This also means the check below only
        // ever considers a lock belonging to somebody else.
        await this.releaseResume(request);

        const unlink = (target) =>
            fs.promises.unlink(target).then(
                () => true,
                () => false,
            );

        // Absent, stale-with-a-dead-owner, or stamped by this process: all
        // report abandoned, and are ours to clear. Anything else is a writer
        // actively downloading right now.
        if (!(await this.#isLockAbandoned(lockPath))) {
            return unlink(filePath);
        }

        const [fileDeleted, partialDeleted] = await Promise.all([
            unlink(filePath),
            unlink(incompletePath),
            unlink(incompletePath + '.json'),
            unlink(lockPath),
        ]);

        return fileDeleted || partialDeleted;
    }

    // TODO add the rest?
    // addAll(requests: RequestInfo[]): Promise<void>;
    // keys(request?: RequestInfo | URL, options?: CacheQueryOptions): Promise<ReadonlyArray<Request>>;
    // match(request: RequestInfo | URL, options?: CacheQueryOptions): Promise<Response | undefined>;
    // matchAll(request?: RequestInfo | URL, options?: CacheQueryOptions): Promise<ReadonlyArray<Response>>;
}
