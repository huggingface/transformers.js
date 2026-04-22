import * as NativeFS from 'native-universal-fs';
import { Buffer } from 'buffer';
import fs from 'node:fs';
import path from 'node:path';

import { FileResponse } from '../hub/FileResponse.js';
import { Random } from '../random.js';
import { apis } from '../../env.js';

// Create a dedicated random instance for generating unique temporary file names
const rng = new Random();

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
     * Checks whether the given request is in the cache.
     * @param {string} request
     * @returns {Promise<FileResponse | undefined>}
     */
    async match(request) {
        let filePath = path.join(this.path, request);
        let file = apis.IS_REACT_NATIVE_ENV ? await FileResponse.create(filePath) : new FileResponse(filePath);
        return file.exists ? file : undefined;
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

        // Include both PID and a random suffix so that concurrent put() call within the same process
        // (e.g., multiple pipelines loading the same file in parallel) each get their own temp file
        // and don't corrupt each other's writes.
        const id = apis.IS_PROCESS_AVAILABLE ? process.pid : Date.now();
        const randomSuffix = rng._int32().toString(36);
        const tmpPath = filePath + `.tmp.${id}.${randomSuffix}`;

        try {
            const contentLength = response.headers.get('Content-Length');
            const total = parseInt(contentLength ?? '0');
            let loaded = 0;

            const reader = response.body.getReader();
            if (apis.IS_REACT_NATIVE_ENV) {
                await NativeFS.mkdir(path.dirname(filePath));
                const chunks = [];
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    chunks.push(value);
                    loaded += value.length;
                    const progress = total ? (loaded / total) * 100 : 0;
                    progress_callback?.({ progress, loaded, total });
                }
                const combined = new Uint8Array(loaded);
                let offset = 0;
                for (const chunk of chunks) { combined.set(chunk, offset); offset += chunk.length; }
                await NativeFS.writeFile(tmpPath, Buffer.from(combined).toString('base64'), 'base64');
                await NativeFS.moveFile(tmpPath, filePath);
            } else {
                await fs.promises.mkdir(path.dirname(filePath), { recursive: true });
                const fileStream = fs.createWriteStream(tmpPath);
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    await new Promise((resolve, reject) => {
                        fileStream.write(value, (err) => err ? reject(err) : resolve());
                    });
                    loaded += value.length;
                    const progress = total ? (loaded / total) * 100 : 0;
                    progress_callback?.({ progress, loaded, total });
                }
                await new Promise((resolve, reject) => {
                    fileStream.close((err) => (err ? reject(err) : resolve()));
                });
                await fs.promises.rename(tmpPath, filePath);
            }
        } catch (error) {
            // Clean up the temp file if an error occurred during download
            try {
                if (apis.IS_REACT_NATIVE_ENV) await NativeFS.unlink(tmpPath); else await fs.promises.unlink(tmpPath);
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
        let filePath = path.join(this.path, request);

        try {
            if (apis.IS_REACT_NATIVE_ENV) await NativeFS.unlink(filePath); else await fs.promises.unlink(filePath);
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
