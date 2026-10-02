/**
 * @file Streaming downloads for React Native.
 *
 * @module utils/hub/download
 */

import * as NativeFS from 'native-universal-fs';
import path from 'node:path';

/**
 * Downloads a file straight to disk, without the payload passing through JavaScript.
 *
 * Model weights routinely run to hundreds of megabytes or more. On React Native the
 * ordinary route -- `fetch` into a `Response`, then write the bytes out -- buffers the
 * entire file in the JS heap (whatwg-fetch has no response stream, so there is nothing to
 * read incrementally), which exhausts the heap on device long before the file is written.
 * The native downloader streams to the target path and only reports progress back.
 *
 * React Native only -- `native-universal-fs` resolves to nothing on other platforms. Node
 * needs no equivalent: its `return_path` loads already stream to disk through `FileCache`.
 *
 * @param {URL|string} fromUrl The URL of the file to download.
 * @param {string} toFile The path to download to.
 * @param {Record<string, string>} [headers] Request headers.
 * @param {(data: {progress: number, loaded: number, total: number}) => void} [progress_callback]
 * Called as bytes land on disk.
 * @returns {Promise<number>} The HTTP status code of the download.
 */
export async function downloadFile(fromUrl, toFile, headers, progress_callback) {
    await NativeFS.mkdir(path.dirname(toFile));

    const { promise } = NativeFS.downloadFile({
        fromUrl: String(fromUrl),
        toFile,
        headers,
        progressInterval: 200,
        progress: ({ contentLength, bytesWritten }) => {
            progress_callback?.({
                progress: contentLength ? (bytesWritten / contentLength) * 100 : 0,
                loaded: bytesWritten,
                total: contentLength,
            });
        },
    });

    const { statusCode } = await promise;
    return statusCode;
}
