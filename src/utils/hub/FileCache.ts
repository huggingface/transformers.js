import fs from 'node:fs';
import path from 'node:path';
import FileResponse from './FileResponse.js';
import { ProgressCallback } from './types.js';

/**
 * File system cache implementation that implements the CacheInterface.
 * Provides `match` and `put` methods compatible with the Web Cache API.
 */
export default class FileCache {
    path: string;

    /**
     * Instantiate a `FileCache` object.
     * @param path The path to the cache directory.
     */
    constructor(path: string) {
        this.path = path;
    }

    /**
     * Checks whether the given request is in the cache.
     * @param request The request to check.
     */
    async match(request: string): Promise<FileResponse> {
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
     * @param [progress_callback] Optional.
     * The function to call with progress updates
     */
    async put(request: string, response: Response, progress_callback: ProgressCallback = undefined): Promise<void> {
        let filePath = path.join(this.path, request);

        try {
            const contentLength = response.headers.get('Content-Length');
            const total = parseInt(contentLength ?? '0');
            let loaded = 0;

            await fs.promises.mkdir(path.dirname(filePath), { recursive: true });
            const fileStream = fs.createWriteStream(filePath);
            const reader = response.body.getReader();

            while (true) {
                const { done, value } = await reader.read();
                if (done) {
                    break;
                }

                await new Promise<void>((resolve, reject) => {
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

            fileStream.close();
        } catch (error) {
            // Clean up the file if an error occurred during download
            try {
                await fs.promises.unlink(filePath);
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
