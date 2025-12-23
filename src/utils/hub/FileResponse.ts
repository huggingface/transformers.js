import fs from 'node:fs';

/**
 * Mapping from file extensions to MIME types.
 */
const CONTENT_TYPE_MAP: Record<string, string> = {
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

export default class FileResponse {
    filePath: string;
    headers: Headers;
    exists: boolean;
    status: number;
    statusText: string;
    body: ReadableStream<Uint8Array> | null;

    /**
     * Creates a new `FileResponse` object.
     * @param filePath - The path to the file
     */
    constructor(filePath: string) {
        this.filePath = filePath;
        this.headers = new Headers();

        this.exists = fs.existsSync(filePath);
        if (this.exists) {
            this.status = 200;
            this.statusText = 'OK';

            let stats = fs.statSync(filePath);
            this.headers.set('content-length', stats.size.toString());

            this.updateContentType();

            const stream = fs.createReadStream(filePath);
            this.body = new ReadableStream({
                start(controller) {
                    stream.on('data', (chunk: Buffer) => controller.enqueue(new Uint8Array(chunk)));
                    stream.on('end', () => controller.close());
                    stream.on('error', (err) => controller.error(err));
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
     */
    updateContentType(): void {
        // Set content-type header based on file extension
        const extension = this.filePath.toString().split('.').pop()?.toLowerCase() ?? '';
        this.headers.set('content-type', CONTENT_TYPE_MAP[extension] ?? 'application/octet-stream');
    }

    /**
     * Clone the current FileResponse object.
     * @returns A new FileResponse object with the same properties as the current object.
     */
    clone(): FileResponse {
        let response = new FileResponse(this.filePath);
        response.exists = this.exists;
        response.status = this.status;
        response.statusText = this.statusText;
        response.headers = new Headers(this.headers);
        return response;
    }

    /**
     * Reads the contents of the file specified by the filePath property and returns a Promise that
     * resolves with an ArrayBuffer containing the file's contents.
     * @returns A Promise that resolves with an ArrayBuffer containing the file's contents.
     * @throws If the file cannot be read.
     */
    async arrayBuffer(): Promise<ArrayBuffer> {
        const data = await fs.promises.readFile(this.filePath);
        return data.buffer as ArrayBuffer;
    }

    /**
     * Reads the contents of the file specified by the filePath property and returns a Promise that
     * resolves with a Blob containing the file's contents.
     * @returns A Promise that resolves with a Blob containing the file's contents.
     * @throws If the file cannot be read.
     */
    async blob(): Promise<Blob> {
        const data = await fs.promises.readFile(this.filePath);
        return new Blob([data], { type: this.headers.get('content-type') ?? 'application/octet-stream' });
    }

    /**
     * Reads the contents of the file specified by the filePath property and returns a Promise that
     * resolves with a string containing the file's contents.
     * @returns A Promise that resolves with a string containing the file's contents.
     * @throws If the file cannot be read.
     */
    async text(): Promise<string> {
        return await fs.promises.readFile(this.filePath, 'utf8');
    }

    /**
     * Reads the contents of the file specified by the filePath property and returns a Promise that
     * resolves with a parsed JavaScript object containing the file's contents.
     *
     * @returns A Promise that resolves with a parsed JavaScript object containing the file's contents.
     * @throws If the file cannot be read.
     */
    async json(): Promise<unknown> {
        return JSON.parse(await this.text());
    }
}
