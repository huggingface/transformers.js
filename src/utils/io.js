import fs from 'node:fs';
import { Readable } from 'node:stream';
import { pipeline as pipe } from 'node:stream/promises';

import { apis } from '../env.js';

/**
 * Save blob file.
 * @param {string} path The path to save the blob to
 * @param {Blob} blob The blob to save
 * @returns {Promise<void>} A promise that resolves when the blob has been saved
 */
export async function saveBlob(path, blob) {
    if (apis.IS_BROWSER_ENV) {
        if (apis.IS_WEBWORKER_ENV) {
            throw new Error('Unable to save a file from a Web Worker.');
        }
        // Convert the canvas content to a data URL
        const dataURL = URL.createObjectURL(blob);

        // Create an anchor element with the data URL as the href attribute
        const downloadLink = document.createElement('a');
        downloadLink.href = dataURL;

        // Set the download attribute to specify the desired filename for the downloaded image
        downloadLink.download = path;

        // Trigger the download
        downloadLink.click();

        // Clean up: remove the anchor element from the DOM
        downloadLink.remove();

        // Revoke the Object URL to free up memory
        URL.revokeObjectURL(dataURL);
    } else if (apis.IS_FS_AVAILABLE) {
        // Convert Blob to a Node.js Readable Stream
        const webStream = blob.stream();
        const nodeStream = Readable.fromWeb(webStream);

        // Create the file write stream
        const fileStream = fs.createWriteStream(path);

        // Pipe the readable stream to the file write stream
        await pipe(nodeStream, fileStream);
    } else {
        throw new Error('Unable to save because filesystem is disabled in this environment.');
    }
}
