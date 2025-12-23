import { ERROR_MAPPING, REPO_ID_REGEX } from './constants.js';
import type FileResponse from './FileResponse.js';
import { ProgressCallback } from './types.js';

/**
 * Joins multiple parts of a path into a single path, while handling leading and trailing slashes.
 *
 * @param parts Multiple parts of a path.
 * @returns A string representing the joined path.
 */
export function pathJoin(...parts: string[]): string {
    // https://stackoverflow.com/a/55142565
    parts = parts.map((part, index) => {
        if (index) {
            part = part.replace(new RegExp('^/'), '');
        }
        if (index !== parts.length - 1) {
            part = part.replace(new RegExp('/$'), '');
        }
        return part;
    });
    return parts.join('/');
}

/**
 * Determines whether the given string is a valid URL.
 * @param string The string to test for validity as an URL.
 * @param protocols A list of valid protocols. If specified, the protocol must be in this list.
 * @param validHosts A list of valid hostnames. If specified, the URL's hostname must be in this list.
 * @returns True if the string is a valid URL, false otherwise.
 */
export function isValidUrl(
    string: string | URL,
    protocols: string[] | null = null,
    validHosts: string[] | null = null,
): boolean {
    let url: URL;
    try {
        url = new URL(string);
    } catch (_) {
        return false;
    }
    if (protocols && !protocols.includes(url.protocol)) {
        return false;
    }
    return !(validHosts && !validHosts.includes(url.hostname));
}

/**
 * Tests whether a string is a valid Hugging Face model ID or not.
 * Adapted from https://github.com/huggingface/huggingface_hub/blob/6378820ebb03f071988a96c7f3268f5bdf8f9449/src/huggingface_hub/utils/_validators.py#L119-L170
 *
 * @param string The string to test
 * @returns True if the string is a valid model ID, false otherwise.
 */
export function isValidHfModelId(string: string): boolean {
    if (!REPO_ID_REGEX.test(string)) return false;
    if (string.includes('..') || string.includes('--')) return false;
    return !(string.endsWith('.git') || string.endsWith('.ipynb'));
}

/**
 * Helper method to handle fatal errors that occur while trying to load a file from the Hugging Face Hub.
 * @param status The HTTP status code of the error.
 * @param remoteURL The URL of the file that could not be loaded.
 * @param fatal Whether to raise an error if the file could not be loaded.
 * @returns Returns `null` if `fatal = true`.
 * @throws If `fatal = false`.
 */
export function handleError(status: number, remoteURL: string, fatal: boolean): null {
    if (!fatal) {
        // File was not loaded correctly, but it is optional.
        // TODO in future, cache the response?
        return null;
    }

    const message = ERROR_MAPPING[status] ?? `Error (${status}) occurred while trying to load file`;
    throw Error(`${message}: "${remoteURL}".`);
}

/**
 * Read and track progress when reading a Response object
 *
 * @param response The Response object to read
 * @param progress_callback The function to call with progress updates
 * @returns A Promise that resolves with the Uint8Array buffer
 */
export async function readResponse(
    response: Response | FileResponse,
    progress_callback: ProgressCallback,
): Promise<Uint8Array> {
    const contentLength = response.headers.get('Content-Length');
    if (contentLength === null) {
        console.warn('Unable to determine content-length from response headers. Will expand buffer when needed.');
    }
    let total = parseInt(contentLength ?? '0');
    let buffer = new Uint8Array(total);
    let loaded = 0;

    const reader = response.body!.getReader();
    async function read(): Promise<void> {
        const { done, value } = await reader.read();
        if (done) return;

        const newLoaded = loaded + value.length;
        if (newLoaded > total) {
            total = newLoaded;

            // Adding the new data will overflow buffer.
            // In this case, we extend the buffer
            const newBuffer = new Uint8Array(total);

            // copy contents
            newBuffer.set(buffer);

            buffer = newBuffer;
        }
        buffer.set(value, loaded);
        loaded = newLoaded;

        const progress = (loaded / total) * 100;

        // Call your function here
        progress_callback({ progress, loaded, total });

        return read();
    }

    // Actually read
    await read();

    return buffer;
}
