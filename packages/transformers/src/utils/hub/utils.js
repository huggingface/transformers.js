import { ERROR_MAPPING, REPO_ID_REGEX } from './constants.js';

/**
 * Joins multiple parts of a path into a single path, while handling leading and trailing slashes.
 *
 * @param {...string} parts Multiple parts of a path.
 * @returns {string} A string representing the joined path.
 */
export function pathJoin(...parts) {
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
 * @param {string|URL} string The string to test for validity as an URL.
 * @param {string[]} [protocols=null] A list of valid protocols. If specified, the protocol must be in this list.
 * @param {string[]} [validHosts=null] A list of valid hostnames. If specified, the URL's hostname must be in this list.
 * @returns {boolean} True if the string is a valid URL, false otherwise.
 */
export function isValidUrl(string, protocols = null, validHosts = null) {
    let url;
    try {
        url = new URL(string);
    } catch (_) {
        return false;
    }
    if (protocols && !protocols.includes(url.protocol)) {
        return false;
    }
    if (validHosts && !validHosts.includes(url.hostname)) {
        return false;
    }
    return true;
}

/**
 * Tests whether a string is a valid Hugging Face model ID or not.
 * Adapted from https://github.com/huggingface/huggingface_hub/blob/6378820ebb03f071988a96c7f3268f5bdf8f9449/src/huggingface_hub/utils/_validators.py#L119-L170
 *
 * @param {string} string The string to test
 * @returns {boolean} True if the string is a valid model ID, false otherwise.
 */
export function isValidHfModelId(string) {
    if (!REPO_ID_REGEX.test(string)) return false;
    if (string.includes('..') || string.includes('--')) return false;
    if (string.endsWith('.git') || string.endsWith('.ipynb')) return false;
    return true;
}

/**
 * Helper method to handle fatal errors that occur while trying to load a file from the Hugging Face Hub.
 * @param {number} status The HTTP status code of the error.
 * @param {string} remoteURL The URL of the file that could not be loaded.
 * @param {boolean} fatal Whether to raise an error if the file could not be loaded.
 * @returns {null} Returns `null` if `fatal = true`.
 * @throws {Error} If `fatal = false`.
 */
export function handleError(status, remoteURL, fatal) {
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
 * @param {Response|import('./files.js').FileResponse} response The Response object to read
 * @param {(data: {progress: number, loaded: number, total: number}) => void} progress_callback The function to call with progress updates
 * @returns {Promise<Uint8Array>} A Promise that resolves with the Uint8Array buffer
 */
export async function readResponse(response, progress_callback) {
    const contentLength = response.headers.get('Content-Length');
    if (contentLength === null) {
        console.warn('Unable to determine content-length from response headers. Will expand buffer when needed.');
    }
    let total = parseInt(contentLength ?? '0');
    let buffer = new Uint8Array(total);
    let loaded = 0;

    const reader = response.body.getReader();
    async function read() {
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
