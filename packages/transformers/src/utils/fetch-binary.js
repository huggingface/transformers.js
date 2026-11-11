/**
 * @file Binary-safe `fetch` for React Native.
 *
 * React Native's `fetch` is backed by a text-oriented networking layer: `Response.body` is
 * not a real stream and `arrayBuffer()` round-trips the payload through a string, which
 * corrupts binary files such as ONNX graphs and `.npy` speaker embeddings. `XMLHttpRequest`
 * with `responseType = 'arraybuffer'` is the supported way to get the raw bytes, so we wrap
 * one in a `fetch`-shaped function and hand it to `env.fetch` on React Native only.
 *
 * This module deliberately imports nothing -- `env.js` consumes it, so any dependency on
 * `env.js` here would be circular.
 *
 * @module utils/fetch-binary
 */

/**
 * Parses a raw XHR header block into a `Headers` object.
 * @param {string} rawHeaders
 * @returns {Headers}
 */
function parseHeaders(rawHeaders) {
    const headers = new Headers();
    // Header values may be folded onto continuation lines; unfold them before splitting.
    const preProcessedHeaders = rawHeaders.replace(/\r?\n[\t ]+/g, ' ');
    for (const line of preProcessedHeaders.split(/\r?\n/)) {
        const parts = line.split(':');
        const key = parts.shift().trim();
        if (key) {
            headers.append(key, parts.join(':').trim());
        }
    }
    return headers;
}

/**
 * Makes a binary-safe request using the XHR API, returning a standard `Response`.
 *
 * @param {string|URL} url
 * @param {RequestInit} [options]
 * @returns {Promise<Response>}
 */
export function fetchBinary(url, options = {}) {
    return new Promise((resolve, reject) => {
        const request = new Request(url, options);
        const xhr = new XMLHttpRequest();

        xhr.onload = () => {
            const headers = parseHeaders(xhr.getAllResponseHeaders() || '');
            // React Native's `Response` polyfill takes the final URL from the init bag,
            // which the standard `ResponseInit` type has no field for -- hence the cast.
            // Without it `response.url` comes back empty and redirects can't be resolved.
            const init = /** @type {ResponseInit} */ ({
                status: xhr.status,
                statusText: xhr.statusText,
                headers,
                url: 'responseURL' in xhr ? xhr.responseURL : (headers.get('x-request-url') ?? ''),
            });
            resolve(new Response(xhr.response, init));
        };

        xhr.onerror = () => reject(new TypeError('Network request failed'));
        xhr.ontimeout = () => reject(new TypeError('Request timeout'));
        xhr.onabort = () => reject(new DOMException('Aborted', 'AbortError'));

        xhr.open(request.method, request.url, true);

        if (request.credentials === 'include') {
            xhr.withCredentials = true;
        } else if (request.credentials === 'omit') {
            xhr.withCredentials = false;
        }

        xhr.responseType = 'arraybuffer';

        request.headers.forEach((value, name) => xhr.setRequestHeader(name, value));

        options.signal?.addEventListener('abort', () => xhr.abort());

        // @ts-ignore -- `_bodyInit` is React Native's internal body holder on `Request`.
        xhr.send(request._bodyInit ?? null);
    });
}
