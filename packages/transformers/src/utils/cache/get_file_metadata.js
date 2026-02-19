/**
 * @file File metadata utilities for cache-aware operations
 */

import { env } from '../../env.js';
import { getCache } from '../cache.js';
import { buildResourcePaths, checkCachedResource, getFetchHeaders, getFile } from '../hub.js';
import { isValidUrl } from '../hub/utils.js';

/**
 * @typedef {import('../hub.js').PretrainedOptions} PretrainedOptions
 */

/**
 * Makes a Range request to get file metadata without downloading full content.
 *
 * We use a Range request (first byte only) instead of HEAD because:
 * 1. Ensures we get the uncompressed file size, which is critical for accurate progress tracking
 * 2. When compression is used, HEAD request content-length reflects compressed size
 * 3. But during actual download, fetch API decompresses transparently and reports uncompressed bytes
 * 4. This mismatch causes progress bar inconsistencies when tracking multiple files
 * 5. Range requests typically aren't compressed, and content-range header shows true uncompressed size
 *
 * @param {URL|string} urlOrPath The URL/path of the file.
 * @returns {Promise<Response|null>} A promise that resolves to a Response object or null if not supported.
 * @private
 */
async function fetch_file_head(urlOrPath) {
    // Range requests only make sense for HTTP URLs
    if (!isValidUrl(urlOrPath, ['http:', 'https:'])) {
        return null;
    }

    const headers = getFetchHeaders(urlOrPath);
    headers.set('Range', 'bytes=0-0');
    return fetch(urlOrPath, { method: 'GET', headers });
}

/**
 * Gets file metadata (size, content-type, etc.) without downloading the full content.
 * Uses Range requests for remote files to be efficient.
 * Can also be used as a lightweight file existence check by checking the `.exists` property.
 *
 * @param {string} path_or_repo_id This can be either:
 * - a string, the *model id* of a model repo on huggingface.co.
 * - a path to a *directory* potentially containing the file.
 * @param {string} filename The name of the file to check.
 * @param {PretrainedOptions} [options] An object containing optional parameters.
 * @returns {Promise<{exists: boolean, size?: number, contentType?: string, fromCache?: boolean}>} A Promise that resolves to file metadata.
 */
export async function get_file_metadata(path_or_repo_id, filename, options = {}) {
    /** @type {import('../cache.js').CacheInterface | null} */
    const cache = await getCache(options?.cache_dir);
    const { localPath, remoteURL, proposedCacheKey, validModelId } = buildResourcePaths(
        path_or_repo_id,
        filename,
        options,
        cache,
    );

    // Check cache first - if cached, we can get metadata from the cached response
    const cachedResponse = await checkCachedResource(cache, localPath, proposedCacheKey);
    if (cachedResponse !== undefined && typeof cachedResponse !== 'string') {
        const size = cachedResponse.headers.get('content-length');
        const contentType = cachedResponse.headers.get('content-type');
        return {
            exists: true,
            size: size ? parseInt(size, 10) : undefined,
            contentType: contentType || undefined,
            fromCache: true,
        };
    }

    // Check local file system
    if (env.allowLocalModels) {
        const isURL = isValidUrl(localPath, ['http:', 'https:']);
        if (!isURL) {
            try {
                const response = await getFile(localPath);
                if (typeof response !== 'string' && response.status !== 404) {
                    const size = response.headers.get('content-length');
                    const contentType = response.headers.get('content-type');
                    return {
                        exists: true,
                        size: size ? parseInt(size, 10) : undefined,
                        contentType: contentType || undefined,
                        fromCache: false,
                    };
                }
            } catch (e) {
                // File doesn't exist locally, continue to remote check
            }
        }
    }

    // Check remote if allowed - use Range request for efficiency
    if (env.allowRemoteModels && !options.local_files_only && validModelId) {
        try {
            // Make a Range request to get metadata without downloading full content
            const rangeResponse = await fetch_file_head(remoteURL);

            if (rangeResponse && rangeResponse.status >= 200 && rangeResponse.status < 300) {
                let size;
                const contentType = rangeResponse.headers.get('content-type');

                // If server supports Range requests, we get a 206 Partial Content response
                // with a content-range header showing the total uncompressed file size
                if (rangeResponse.status === 206) {
                    const contentRange = rangeResponse.headers.get('content-range');
                    if (contentRange) {
                        // Format: "bytes 0-0/12345" where 12345 is the total uncompressed size
                        const match = contentRange.match(/bytes \d+-\d+\/(\d+)/);
                        if (match) {
                            size = parseInt(match[1], 10);
                        }
                    }
                } else if (rangeResponse.status === 200) {
                    // Server doesn't support Range requests and returned the full file (200 OK)
                    // Cancel the response body immediately to avoid downloading the entire file
                    try {
                        await rangeResponse.body?.cancel();
                    } catch (cancelError) {
                        // Ignore cancellation errors - body might already be consumed or not cancellable
                    }
                }

                // Fallback to content-length if content-range is not available (200 OK response)
                // This can happen if server doesn't support Range requests
                if (size === undefined) {
                    const contentLength = rangeResponse.headers.get('content-length');
                    size = contentLength ? parseInt(contentLength, 10) : undefined;
                }

                return {
                    exists: true,
                    size,
                    contentType: contentType || undefined,
                    fromCache: false,
                };
            }
        } catch (e) {
            // Range request failed, fall back to regular GET
            try {
                const response = await getFile(remoteURL);
                if (typeof response !== 'string' && response.status >= 200 && response.status < 300) {
                    const size = response.headers.get('content-length');
                    const contentType = response.headers.get('content-type');
                    return {
                        exists: true,
                        size: size ? parseInt(size, 10) : undefined,
                        contentType: contentType || undefined,
                        fromCache: false,
                    };
                }
            } catch (e2) {
                // Ignore
            }
        }
    }

    return { exists: false, fromCache: false };
}
