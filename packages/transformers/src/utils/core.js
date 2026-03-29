/**
 * @file Core utility functions/classes for Transformers.js.
 *
 * These are only used internally, meaning an end-user shouldn't
 * need to access anything here.
 *
 * @module utils/core
 */

/**
 * @typedef {Object} InitiateProgressInfo
 * @property {'initiate'} status
 * @property {string} name The model id or directory path.
 * @property {string} file The name of the file.
 */

/**
 * @typedef {Object} DownloadProgressInfo
 * @property {'download'} status
 * @property {string} name The model id or directory path.
 * @property {string} file The name of the file.
 */

/**
 * @typedef {Object} ProgressStatusInfo
 * @property {'progress'} status
 * @property {string} name The model id or directory path.
 * @property {string} file The name of the file.
 * @property {number} progress A number between 0 and 100.
 * @property {number} loaded The number of bytes loaded.
 * @property {number} total The total number of bytes to be loaded.
 */

/**
 * @typedef {Object} FileLoadingProgress
 * @property {number} loaded The number of bytes loaded for this file.
 * @property {number} total The total number of bytes for this file.
 */

/**
 * @typedef {Record<string, FileLoadingProgress>} FilesLoadingMap
 * A mapping of file names to their loading progress. Each key is a file path and each value contains
 * the loaded and total bytes for that file.
 */

/**
 * @typedef {Object} TotalProgressInfo
 * @property {'progress_total'} status
 * @property {string} name The model id or directory path.
 * @property {number} progress A number between 0 and 100.
 * @property {number} loaded The number of bytes loaded.
 * @property {number} total The total number of bytes to be loaded.
 * @property {FilesLoadingMap} files A mapping of file names to their loading progress.
 */

/**
 * @typedef {Object} DoneProgressInfo
 * @property {'done'} status
 * @property {string} name The model id or directory path.
 * @property {string} file The name of the file.
 */

/**
 * @typedef {Object} ReadyProgressInfo
 * @property {'ready'} status
 * @property {string} task The loaded task.
 * @property {string} model The loaded model.
 */

/**
 * @typedef {InitiateProgressInfo | DownloadProgressInfo | ProgressStatusInfo | DoneProgressInfo | ReadyProgressInfo | TotalProgressInfo} ProgressInfo
 */

/**
 * A callback function that is called with progress information.
 * @callback ProgressCallback
 * @param {ProgressInfo} progressInfo
 * @returns {void}
 */

/**
 * Helper function to dispatch progress callbacks.
 *
 * @param {ProgressCallback | null | undefined} progress_callback The progress callback function to dispatch.
 * @param {ProgressInfo} data The data to pass to the progress callback function.
 * @returns {void}
 * @private
 */
export function dispatchCallback(progress_callback, data) {
    if (progress_callback) progress_callback(data);
}

const PROGRESS_CALLBACK_WRAPPERS = new WeakMap();

/**
 * Wrapper class for progress callbacks.
 *
 * Keeps callback wrapping metadata outside function objects, allowing callers
 * to check whether a callback is already wrapped.
 */
export class ProgressCallbackWrapper {
    /**
     * @param {ProgressCallback} callback The original callback.
     * @param {(info: ProgressInfo, callback: ProgressCallback) => void} handler
     * A handler that receives each progress event and the original callback.
     */
    constructor(callback, handler) {
        this.callback = callback;
        this.handler = handler;
        this.wrapped = (info) => this.handler(info, this.callback);

        PROGRESS_CALLBACK_WRAPPERS.set(this.wrapped, this);
    }

    /**
     * Checks whether a callback has been created by this wrapper class.
     *
     * @param {ProgressCallback | null | undefined} callback
     * @returns {boolean}
     */
    static isWrapped(callback) {
        if (!callback) return false;
        return PROGRESS_CALLBACK_WRAPPERS.get(callback) instanceof ProgressCallbackWrapper;
    }
}

/**
 * Creates a progress callback wrapper that emits `progress_total` events
 * based on the aggregate state of `files_loading`.
 * The wrapper also lets nested loaders detect wrapped callbacks via
 * `ProgressCallbackWrapper.isWrapped(...)` so we avoid double-wrapping and
 * duplicate `progress_total` events without mutating function objects.
 *
 * @param {ProgressCallback} progress_callback The callback to wrap.
 * @param {FilesLoadingMap} files_loading Mutable map storing per-file progress.
 * @returns {ProgressCallback} Wrapped callback.
 */
export function createTotalProgressCallback(progress_callback, files_loading) {
    const wrapper = new ProgressCallbackWrapper(progress_callback, (info, callback) => {
        if (info.status === 'progress') {
            files_loading[info.file] = {
                loaded: info.loaded,
                total: info.total,
            };

            const loaded = Object.values(files_loading).reduce((acc, curr) => acc + curr.loaded, 0);
            const total = Object.values(files_loading).reduce((acc, curr) => acc + curr.total, 0);
            const progress = total > 0 ? (loaded / total) * 100 : 0;

            callback({
                status: 'progress_total',
                name: info.name,
                progress,
                loaded,
                total,
                files: structuredClone(files_loading),
            });
        }
        callback(info);
    });

    return wrapper.wrapped;
}

/**
 * Reverses the keys and values of an object.
 *
 * @param {Object} data The object to reverse.
 * @returns {Object} The reversed object.
 * @see https://ultimatecourses.com/blog/reverse-object-keys-and-values-in-javascript
 */
export function reverseDictionary(data) {
    // https://ultimatecourses.com/blog/reverse-object-keys-and-values-in-javascript
    return Object.fromEntries(Object.entries(data).map(([key, value]) => [value, key]));
}

/**
 * Escapes regular expression special characters from a string by replacing them with their escaped counterparts.
 *
 * @param {string} string The string to escape.
 * @returns {string} The escaped string.
 */
export function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); // $& means the whole matched string
}

/**
 * Check if a value is a typed array.
 * @param {*} val The value to check.
 * @returns {boolean} True if the value is a `TypedArray`, false otherwise.
 *
 * Adapted from https://stackoverflow.com/a/71091338/13989043
 */
export function isTypedArray(val) {
    return val?.prototype?.__proto__?.constructor?.name === 'TypedArray';
}

/**
 * Check if a value is an integer.
 * @param {*} x The value to check.
 * @returns {boolean} True if the value is a string, false otherwise.
 */
export function isIntegralNumber(x) {
    return Number.isInteger(x) || typeof x === 'bigint';
}

/**
 * Determine if a provided width or height is nullish.
 * @param {*} x The value to check.
 * @returns {boolean} True if the value is `null`, `undefined` or `-1`, false otherwise.
 */
export function isNullishDimension(x) {
    return x === null || x === undefined || x === -1;
}

/**
 * Calculates the dimensions of a nested array.
 *
 * @param {any[]} arr The nested array to calculate dimensions for.
 * @returns {number[]} An array containing the dimensions of the input array.
 */
export function calculateDimensions(arr) {
    const dimensions = [];
    let current = arr;
    while (Array.isArray(current)) {
        dimensions.push(current.length);
        current = current[0];
    }
    return dimensions;
}

/**
 * Replicate python's .pop() method for objects.
 * @param {Object} obj The object to pop from.
 * @param {string} key The key to pop.
 * @param {*} defaultValue The default value to return if the key does not exist.
 * @returns {*} The value of the popped key.
 * @throws {Error} If the key does not exist and no default value is provided.
 */
export function pop(obj, key, defaultValue = undefined) {
    const value = obj[key];
    if (value !== undefined) {
        delete obj[key];
        return value;
    }
    if (defaultValue === undefined) {
        throw Error(`Key ${key} does not exist in object.`);
    }
    return defaultValue;
}

/**
 * Efficiently merge arrays, creating a new copy.
 * Adapted from https://stackoverflow.com/a/6768642/13989043
 * @param  {any[]} arrs Arrays to merge.
 * @returns {any[]} The merged array.
 */
export function mergeArrays(...arrs) {
    return Array.prototype.concat.apply([], arrs);
}

/**
 * Compute the Cartesian product of given arrays
 * @param {...any[]} a Arrays to compute the product
 * @returns {any[]} Returns the computed Cartesian product as an array
 * @private
 */
export function product(...a) {
    // Cartesian product of items
    // Adapted from https://stackoverflow.com/a/43053803
    return a.reduce((a, b) => a.flatMap((d) => b.map((e) => [d, e])));
}

/**
 * Calculates the index offset for a given index and window size.
 * @param {number} i The index.
 * @param {number} w The window size.
 * @returns {number} The index offset.
 */
export function calculateReflectOffset(i, w) {
    return Math.abs(((i + w) % (2 * w)) - w);
}

/**
 *
 * @param {Object} o
 * @param {string[]} props
 * @returns {Object}
 */
export function pick(o, props) {
    return Object.assign(
        {},
        ...props.map((prop) => {
            if (o[prop] !== undefined) {
                return { [prop]: o[prop] };
            }
        }),
    );
}

/**
 * Calculate the length of a string, taking multi-byte characters into account.
 * This mimics the behavior of Python's `len` function.
 * @param {string} s The string to calculate the length of.
 * @returns {number} The length of the string.
 */
export function len(s) {
    let length = 0;
    for (const c of s) ++length;
    return length;
}

/**
 * Count the occurrences of a value in an array or string.
 * This mimics the behavior of Python's `count` method.
 * @param {any[]|string} arr The array or string to search.
 * @param {any} value The value to count.
 */
export function count(arr, value) {
    let count = 0;
    for (const v of arr) {
        if (v === value) ++count;
    }
    return count;
}
