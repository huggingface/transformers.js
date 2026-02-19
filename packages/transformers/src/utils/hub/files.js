import fs from 'node:fs';
import path from 'node:path';

/**
 * Mapping from file extensions to MIME types.
 */
const CONTENT_TYPE_MAP = {
    txt: 'text/plain',
    html: 'text/html',
    css: 'text/css',
    js: 'text/javascript',
    json: 'application/json',
    png: 'image/png',
    jpg: 'image/jpeg',
    jpeg: 'image/jpeg',
    gif: 'image/gif',
};

export class FileResponse {
    /**
     * Creates a new `FileResponse` object.
     * @param {string} filePath
     * @param {Object} options Additional options for creating the file response.
     * @param {AbortSignal|null} [options.abort_signal=null] An optional AbortSignal to cancel the request.
     */
    constructor(filePath, { abort_signal = null } = {}) {
        this.filePath = filePath;
        this.headers = new Headers();
        this.abort_signal = abort_signal;

        this.exists = fs.existsSync(filePath);
        if (this.exists) {
            this.status = 200;
            this.statusText = 'OK';

            let stats = fs.statSync(filePath);
            this.headers.set('content-length', stats.size.toString());

            this.updateContentType();

            const stream = fs.createReadStream(filePath, { signal: abort_signal });
            this.body = new ReadableStream({
                start(controller) {
                    stream.on('data', (chunk) => controller.enqueue(chunk));
                    stream.on('end', () => controller.close());
                    stream.on('error', (err) => controller.error(err));
                    abort_signal?.addEventListener('abort', () => {
                        controller.error(new Error('Request aborted'));
                    });
                },
                cancel() {
                    stream.destroy();
                },
            });
        } else {
            this.status = 404;
            this.statusText = 'Not Found';
            this.body = null;
        }
    }

    /**
     * Updates the 'content-type' header property of the response based on the extension of
     * the file specified by the filePath property of the current object.
     * @returns {void}
     */
    updateContentType() {
        // Set content-type header based on file extension
        const extension = this.filePath.toString().split('.').pop().toLowerCase();
        this.headers.set('content-type', CONTENT_TYPE_MAP[extension] ?? 'application/octet-stream');
    }

    /**
     * Clone the current FileResponse object.
     * @returns {FileResponse} A new FileResponse object with the same properties as the current object.
     */
    clone() {
        let response = new FileResponse(this.filePath, { abort_signal: this.abort_signal });
        response.exists = this.exists;
        response.status = this.status;
        response.statusText = this.statusText;
        response.headers = new Headers(this.headers);
        return response;
    }

    /**
     * Reads the contents of the file specified by the filePath property and returns a Promise that
     * resolves with an ArrayBuffer containing the file's contents.
     * @returns {Promise<ArrayBuffer>} A Promise that resolves with an ArrayBuffer containing the file's contents.
     * @throws {Error} If the file cannot be read.
     */
    async arrayBuffer() {
        const data = await fs.promises.readFile(this.filePath);
        return /** @type {ArrayBuffer} */ (data.buffer);
    }

    /**
     * Reads the contents of the file specified by the filePath property and returns a Promise that
     * resolves with a Blob containing the file's contents.
     * @returns {Promise<Blob>} A Promise that resolves with a Blob containing the file's contents.
     * @throws {Error} If the file cannot be read.
     */
    async blob() {
        const data = await fs.promises.readFile(this.filePath);
        return new Blob([/** @type {any} */ (data)], { type: this.headers.get('content-type') });
    }

    /**
     * Reads the contents of the file specified by the filePath property and returns a Promise that
     * resolves with a string containing the file's contents.
     * @returns {Promise<string>} A Promise that resolves with a string containing the file's contents.
     * @throws {Error} If the file cannot be read.
     */
    async text() {
        return await fs.promises.readFile(this.filePath, 'utf8');
    }

    /**
     * Reads the contents of the file specified by the filePath property and returns a Promise that
     * resolves with a parsed JavaScript object containing the file's contents.
     *
     * @returns {Promise<Object>} A Promise that resolves with a parsed JavaScript object containing the file's contents.
     * @throws {Error} If the file cannot be read.
     */
    async json() {
        return JSON.parse(await this.text());
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
     * @param {Object} options Additional options for adding the response to the cache.
     * @param {AbortSignal|null} [options.abort_signal=null] An optional AbortSignal to cancel the request.
     * The function to call with progress updates
     * @returns {Promise<void>}
     */
    async put(request, response, progress_callback = undefined, { abort_signal = null } = {}) {
        let filePath = path.join(this.path, request);

        try {
            const contentLength = response.headers.get('Content-Length');
            const total = parseInt(contentLength ?? '0');
            let loaded = 0;

            await fs.promises.mkdir(path.dirname(filePath), { recursive: true });
            const fileStream = fs.createWriteStream(filePath);
            const reader = response.body.getReader();

            while (true) {
                // Check if the request has been aborted
                if (abort_signal?.aborted) {
                    reader.cancel();
                    fileStream.destroy();
                    throw new Error('Request aborted');
                }

                const { done, value } = await reader.read();
                if (done) {
                    break;
                }

                // Check again after reading
                if (abort_signal?.aborted) {
                    reader.cancel();
                    fileStream.destroy();
                    throw new Error('Request aborted');
                }

                await new Promise((resolve, reject) => {
                    // Set up abort listener
                    const abortHandler = () => {
                        reader.cancel();
                        fileStream.destroy();
                        reject(new Error('Request aborted'));
                    };

                    if (abort_signal) {
                        abort_signal.addEventListener('abort', abortHandler);
                    }

                    // Write the value to the file stream
                    fileStream.write(value, (err) => {
                        if (abort_signal) {
                            abort_signal.removeEventListener('abort', abortHandler);
                        }
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
