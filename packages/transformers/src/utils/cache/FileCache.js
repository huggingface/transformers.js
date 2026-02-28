import fs from 'node:fs';
import path from 'node:path';

import { FileResponse } from '../hub/FileResponse.js';

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
        let file = new FileResponse(filePath);

        if (file.exists) {
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
        let filePath = path.join(this.path, request);
        let tmpPath = filePath + `.tmp.${process.pid}`;

        try {
            const contentLength = response.headers.get('Content-Length');
            const total = parseInt(contentLength ?? '0');
            let loaded = 0;

            await fs.promises.mkdir(path.dirname(filePath), { recursive: true });
            const fileStream = fs.createWriteStream(tmpPath);
            const reader = response.body.getReader();

            while (true) {
                const { done, value } = await reader.read();
                if (done) {
                    break;
                }

                await new Promise((resolve, reject) => {
                    fileStream.write(value, (err) => {
                        if (err) {
                            reject(err);
                            return;
                        }
                        resolve();
                    });
                });

                loaded += value.length;
                const progress = total ? (loaded / total) * 100 : 0;

                progress_callback?.({ progress, loaded, total });
            }

            await new Promise((resolve, reject) => {
                fileStream.close((err) => (err ? reject(err) : resolve()));
            });

            // Atomically move the completed temp file to the final path so that
            // other processes never observe a partially-written file.
            await fs.promises.rename(tmpPath, filePath);
        } catch (error) {
            // Clean up the temp file if an error occurred during download
            try {
                await fs.promises.unlink(tmpPath);
            } catch {}
            throw error;
        }
    }

    // TODO add the rest?
    // addAll(requests: RequestInfo[]): Promise<void>;
    // delete(request: RequestInfo | URL, options?: CacheQueryOptions): Promise<boolean>;
    // keys(request?: RequestInfo | URL, options?: CacheQueryOptions): Promise<ReadonlyArray<Request>>;
    // match(request: RequestInfo | URL, options?: CacheQueryOptions): Promise<Response | undefined>;
    // matchAll(request?: RequestInfo | URL, options?: CacheQueryOptions): Promise<ReadonlyArray<Response>>;
}
