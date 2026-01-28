/**
 * @file Utility functions to interact with the Hugging Face Hub (https://huggingface.co/models)
 *
 * @module utils/hub
 */

import { apis, env } from '../env.js';
import { dispatchCallback } from './core.js';
import { FileResponse, FileCache } from './hub/files.js';
import { handleError, isValidUrl, pathJoin, isValidHfModelId, readResponse } from './hub/utils.js';
import { getCache, tryCache } from './cache.js';

export { MAX_EXTERNAL_DATA_CHUNKS } from './hub/constants.js';

/**
 * @typedef {boolean|number} ExternalData
 * Specifies whether to load the model using the external data format.
 * - `false`: Do not use external data format
 * - `true`: Use external data format with 1 chunk
 * - `number`: Use external data format with the specified number of chunks
 */

/**
 * @typedef {Object} PretrainedOptions Options for loading a pretrained model.
 * @property {import('./core.js').ProgressCallback} [progress_callback=null] If specified, this function will be called during model construction, to provide the user with progress updates.
 * @property {import('../configs.js').PretrainedConfig} [config=null] Configuration for the model to use instead of an automatically loaded configuration. Configuration can be automatically loaded when:
 * - The model is a model provided by the library (loaded with the *model id* string of a pretrained model).
 * - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a configuration JSON file named *config.json* is found in the directory.
 * @property {string} [cache_dir=null] Path to a directory in which a downloaded pretrained model configuration should be cached if the standard cache should not be used.
 * @property {boolean} [local_files_only=false] Whether or not to only look at local files (e.g., not try downloading the model).
 * @property {string} [revision='main'] The specific model version to use. It can be a branch name, a tag name, or a commit id,
 * since we use a git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier allowed by git.
 * NOTE: This setting is ignored for local requests.
 */

/**
 * @typedef {Object} ModelSpecificPretrainedOptions Options for loading a pretrained model.
 * @property {string} [subfolder='onnx'] In case the relevant files are located inside a subfolder of the model repo on huggingface.co,
 * you can specify the folder name here.
 * @property {string} [model_file_name=null] If specified, load the model with this name (excluding the .onnx suffix). Currently only valid for encoder- or decoder-only models.
 * @property {import("./devices.js").DeviceType|Record<string, import("./devices.js").DeviceType>} [device=null] The device to run the model on. If not specified, the device will be chosen from the environment settings.
 * @property {import("./dtypes.js").DataType|Record<string, import("./dtypes.js").DataType>} [dtype=null] The data type to use for the model. If not specified, the data type will be chosen from the environment settings.
 * @property {ExternalData|Record<string, ExternalData>} [use_external_data_format=false] Whether to load the model using the external data format (used for models >= 2GB in size).
 * @property {import('onnxruntime-common').InferenceSession.SessionOptions} [session_options] (Optional) User-specified session options passed to the runtime. If not provided, suitable defaults will be chosen.
 */

/**
 * @typedef {PretrainedOptions & ModelSpecificPretrainedOptions} PretrainedModelOptions Options for loading a pretrained model.
 */

/**
 * Helper function to get a file, using either the Fetch API or FileSystem API.
 *
 * @param {URL|string} urlOrPath The URL/path of the file to get.
 * @returns {Promise<FileResponse|Response>} A promise that resolves to a FileResponse object (if the file is retrieved using the FileSystem API), or a Response object (if the file is retrieved using the Fetch API).
 */
export async function getFile(urlOrPath) {
    if (env.useFS && !isValidUrl(urlOrPath, ['http:', 'https:', 'blob:'])) {
        return new FileResponse(
            urlOrPath instanceof URL
                ? urlOrPath.protocol === 'file:'
                    ? urlOrPath.pathname
                    : urlOrPath.toString()
                : urlOrPath,
        );
    } else if (typeof process !== 'undefined' && process?.release?.name === 'node') {
        const IS_CI = !!process.env?.TESTING_REMOTELY;
        const version = env.version;

        const headers = new Headers();
        headers.set('User-Agent', `transformers.js/${version}; is_ci/${IS_CI};`);

        // Check whether we are making a request to the Hugging Face Hub.
        const isHFURL = isValidUrl(urlOrPath, ['http:', 'https:'], ['huggingface.co', 'hf.co']);
        if (isHFURL) {
            // If an access token is present in the environment variables,
            // we add it to the request headers.
            // NOTE: We keep `HF_ACCESS_TOKEN` for backwards compatibility (as a fallback).
            const token = process.env?.HF_TOKEN ?? process.env?.HF_ACCESS_TOKEN;
            if (token) {
                headers.set('Authorization', `Bearer ${token}`);
            }
        }
        return fetch(urlOrPath, { headers });
    } else {
        // Running in a browser-environment, so we use default headers
        // NOTE: We do not allow passing authorization headers in the browser,
        // since this would require exposing the token to the client.
        return fetch(urlOrPath);
    }
}

/**
 * Builds the resource paths and URLs for a model file.
 * Can be used to get the resource URL or path without loading the file.
 *
 * @param {string} path_or_repo_id This can be either:
 * - a string, the *model id* of a model repo on huggingface.co.
 * - a path to a *directory* potentially containing the file.
 * @param {string} filename The name of the file to locate.
 * @param {PretrainedOptions} [options] An object containing optional parameters.
 * @param {import('./cache.js').CacheInterface | null} [cache] The cache instance to use for determining cache keys.
 * @returns {{ requestURL: string, localPath: string, remoteURL: string, proposedCacheKey: string, validModelId: boolean }}
 * An object containing all the paths and URLs for the resource.
 */
function buildResourcePaths(path_or_repo_id, filename, options = {}, cache = null) {
    const revision = options.revision ?? 'main';
    const requestURL = pathJoin(path_or_repo_id, filename);

    const validModelId = isValidHfModelId(path_or_repo_id);
    const localPath = validModelId ? pathJoin(env.localModelPath, requestURL) : requestURL;
    const remoteURL = pathJoin(
        env.remoteHost,
        env.remotePathTemplate
            .replaceAll('{model}', path_or_repo_id)
            .replaceAll('{revision}', encodeURIComponent(revision)),
        filename,
    );

    const proposedCacheKey =
        cache instanceof FileCache
            ? // Choose cache key for filesystem cache
              // When using the main revision (default), we use the request URL as the cache key.
              // If a specific revision is requested, we account for this in the cache key.
              revision === 'main'
                ? requestURL
                : pathJoin(path_or_repo_id, revision, filename)
            : remoteURL;

    return {
        requestURL,
        localPath,
        remoteURL,
        proposedCacheKey,
        validModelId,
    };
}

/**
 * Checks if a resource exists in cache.
 *
 * @param {import('./cache.js').CacheInterface | null} cache The cache instance to check.
 * @param {string} localPath The local path to try first.
 * @param {string} proposedCacheKey The proposed cache key to try second.
 * @returns {Promise<Response|import('./hub/files.js').FileResponse|undefined|string>}
 * The cached response if found, undefined otherwise.
 */
export async function checkCachedResource(cache, localPath, proposedCacheKey) {
    if (!cache) {
        return undefined;
    }

    // A caching system is available, so we try to get the file from it.
    //  1. We first try to get from cache using the local path. In some environments (like deno),
    //     non-URL cache keys are not allowed. In these cases, `response` will be undefined.
    //  2. If no response is found, we try to get from cache using the remote URL or file system cache.
    return await tryCache(cache, localPath, proposedCacheKey);
}

/**
 * Stores a resource in the cache.
 *
 * @param {string} path_or_repo_id The path or repo ID of the model.
 * @param {string} filename The name of the file to cache.
 * @param {import('./cache.js').CacheInterface} cache The cache instance to store in.
 * @param {string} cacheKey The cache key to use.
 * @param {Response|import('./hub/files.js').FileResponse} response The response to cache.
 * @param {Uint8Array} [result] The result buffer if already read.
 * @param {PretrainedOptions} [options] Options containing progress callback and context for progress updates.
 * @returns {Promise<void>}
 */
export async function storeCachedResource(path_or_repo_id, filename, cache, cacheKey, response, result, options = {}) {
    // Check again whether request is in cache. If not, we add the response to the cache
    if ((await cache.match(cacheKey)) !== undefined) {
        return;
    }

    if (!result) {
        // We haven't yet read the response body, so we need to do so now.
        // Ensure progress updates include consistent metadata.
        const wrapped_progress = options.progress_callback
            ? (data) =>
                  dispatchCallback(options.progress_callback, {
                      status: 'progress',
                      name: path_or_repo_id,
                      file: filename,
                      ...data,
                  })
            : undefined;
        await cache.put(cacheKey, /** @type {Response} */ (response), wrapped_progress);
    } else if (typeof response !== 'string') {
        // NOTE: We use `new Response(buffer, ...)` instead of `response.clone()` to handle LFS files
        await cache
            .put(
                cacheKey,
                new Response(/** @type {any} */ (result), {
                    headers: response.headers,
                }),
            )
            .catch((err) => {
                // Do not crash if unable to add to cache (e.g., QuotaExceededError).
                // Rather, log a warning and proceed with execution.
                console.warn(`Unable to add response to browser cache: ${err}.`);
            });
    }
}

/**
 * Loads a resource file from local or remote sources.
 *
 * @param {string} path_or_repo_id This can be either:
 * - a string, the *model id* of a model repo on huggingface.co.
 * - a path to a *directory* potentially containing the file.
 * @param {string} filename The name of the file to locate.
 * @param {boolean} [fatal=true] Whether to throw an error if the file is not found.
 * @param {PretrainedOptions} [options] An object containing optional parameters.
 * @param {boolean} [return_path=false] Whether to return the path of the file instead of the file content.
 * @param {import('./cache.js').CacheInterface | null} [cache] The cache instance to use.
 *
 * @throws Will throw an error if the file is not found and `fatal` is true.
 * @returns {Promise<string|Uint8Array|null>} A Promise that resolves with the file content as a Uint8Array if `return_path` is false, or the file path as a string if `return_path` is true.
 */
export async function loadResourceFile(
    path_or_repo_id,
    filename,
    fatal = true,
    options = {},
    return_path = false,
    cache = null,
) {
    const { requestURL, localPath, remoteURL, proposedCacheKey, validModelId } = buildResourcePaths(
        path_or_repo_id,
        filename,
        options,
        cache,
    );

    /** @type {string} */
    let cacheKey;

    // Whether to cache the final response in the end.
    let toCacheResponse = false;

    /** @type {Response|import('./hub/files.js').FileResponse|undefined|string} */
    let response;

    // Check cache
    response = await checkCachedResource(cache, localPath, proposedCacheKey);

    const cacheHit = response !== undefined;
    if (!cacheHit) {
        // Caching not available, or file is not cached, so we perform the request

        if (env.allowLocalModels) {
            // Accessing local models is enabled, so we try to get the file locally.
            // If request is a valid HTTP URL, we skip the local file check. Otherwise, we try to get the file locally.
            const isURL = isValidUrl(requestURL, ['http:', 'https:']);
            if (!isURL) {
                try {
                    response = await getFile(localPath);
                    cacheKey = localPath; // Update the cache key to be the local path
                } catch (e) {
                    // Something went wrong while trying to get the file locally.
                    // NOTE: error handling is done in the next step (since `response` will be undefined)
                    console.warn(`Unable to load from local path "${localPath}": "${e}"`);
                }
            } else if (options.local_files_only) {
                throw new Error(`\`local_files_only=true\`, but attempted to load a remote file from: ${requestURL}.`);
            } else if (!env.allowRemoteModels) {
                throw new Error(
                    `\`env.allowRemoteModels=false\`, but attempted to load a remote file from: ${requestURL}.`,
                );
            }
        }

        if (response === undefined || (typeof response !== 'string' && response.status === 404)) {
            // File not found locally. This means either:
            // - The user has disabled local file access (`env.allowLocalModels=false`)
            // - the path is a valid HTTP url (`response === undefined`)
            // - the path is not a valid HTTP url and the file is not present on the file system or local server (`response.status === 404`)

            if (options.local_files_only || !env.allowRemoteModels) {
                // User requested local files only, but the file is not found locally.
                if (fatal) {
                    throw Error(
                        `\`local_files_only=true\` or \`env.allowRemoteModels=false\` and file was not found locally at "${localPath}".`,
                    );
                } else {
                    // File not found, but this file is optional.
                    // TODO in future, cache the response?
                    return null;
                }
            }
            if (!validModelId) {
                // Before making any requests to the remote server, we check if the model ID is valid.
                // This prevents unnecessary network requests for invalid model IDs.
                throw Error(
                    `Local file missing at "${localPath}" and download aborted due to invalid model ID "${path_or_repo_id}".`,
                );
            }

            // File not found locally, so we try to download it from the remote server
            response = await getFile(remoteURL);

            if (response.status !== 200) {
                return handleError(response.status, remoteURL, fatal);
            }

            // Success! We use the proposed cache key from earlier
            cacheKey = proposedCacheKey;
        }

        // Only cache the response if:
        toCacheResponse =
            cache && // 1. A caching system is available
            typeof Response !== 'undefined' && // 2. `Response` is defined (i.e., we are in a browser-like environment)
            response instanceof Response && // 3. result is a `Response` object (i.e., not a `FileResponse`)
            response.status === 200; // 4. request was successful (status code 200)
    }

    // Start downloading
    dispatchCallback(options.progress_callback, {
        status: 'download',
        name: path_or_repo_id,
        file: filename,
    });

    let result;
    if (!(apis.IS_NODE_ENV && return_path)) {
        /** @type {Uint8Array} */
        let buffer;

        if (typeof response !== 'string') {
            if (!options.progress_callback) {
                // If no progress callback is specified, we can use the `.arrayBuffer()`
                // method to read the response.
                buffer = new Uint8Array(await response.arrayBuffer());
            } else if (
                cacheHit && // The item is being read from the cache
                typeof navigator !== 'undefined' &&
                /firefox/i.test(navigator.userAgent) // We are in Firefox
            ) {
                // Due to bug in Firefox, we cannot display progress when loading from cache.
                // Fortunately, since this should be instantaneous, this should not impact users too much.
                buffer = new Uint8Array(await response.arrayBuffer());

                // For completeness, we still fire the final progress callback
                dispatchCallback(options.progress_callback, {
                    status: 'progress',
                    name: path_or_repo_id,
                    file: filename,
                    progress: 100,
                    loaded: buffer.length,
                    total: buffer.length,
                });
            } else {
                buffer = await readResponse(response, (data) => {
                    dispatchCallback(options.progress_callback, {
                        status: 'progress',
                        name: path_or_repo_id,
                        file: filename,
                        ...data,
                    });
                });
            }
        }
        result = buffer;
    }

    if (
        // Only cache web responses
        // i.e., do not cache FileResponses (prevents duplication)
        toCacheResponse &&
        cacheKey &&
        typeof response !== 'string'
    ) {
        await storeCachedResource(path_or_repo_id, filename, cache, cacheKey, response, result, options);
    }

    dispatchCallback(options.progress_callback, {
        status: 'done',
        name: path_or_repo_id,
        file: filename,
    });

    if (result) {
        if (!apis.IS_NODE_ENV && return_path) {
            throw new Error('Cannot return path in a browser environment.');
        }
        return result;
    }
    if (response instanceof FileResponse) {
        return response.filePath;
    }

    // Otherwise, return the cached response (most likely a `FileResponse`).
    // NOTE: A custom cache may return a Response, or a string (file path)
    const cachedResponse = await cache?.match(cacheKey);
    if (cachedResponse instanceof FileResponse) {
        return cachedResponse.filePath;
    } else if (cachedResponse instanceof Response) {
        return new Uint8Array(await cachedResponse.arrayBuffer());
    } else if (typeof cachedResponse === 'string') {
        return cachedResponse;
    }

    throw new Error('Unable to get model file path or buffer.');
}

/**
 * Retrieves a file from either a remote URL using the Fetch API or from the local file system using the FileSystem API.
 * If the filesystem is available and `env.useCache = true`, the file will be downloaded and cached.
 *
 * @param {string} path_or_repo_id This can be either:
 * - a string, the *model id* of a model repo on huggingface.co.
 * - a path to a *directory* potentially containing the file.
 * @param {string} filename The name of the file to locate in `path_or_repo`.
 * @param {boolean} [fatal=true] Whether to throw an error if the file is not found.
 * @param {PretrainedOptions} [options] An object containing optional parameters.
 * @param {boolean} [return_path=false] Whether to return the path of the file instead of the file content.
 *
 * @throws Will throw an error if the file is not found and `fatal` is true.
 * @returns {Promise<string|Uint8Array>} A Promise that resolves with the file content as a Uint8Array if `return_path` is false, or the file path as a string if `return_path` is true.
 */
export async function getModelFile(path_or_repo_id, filename, fatal = true, options = {}, return_path = false) {
    if (!env.allowLocalModels) {
        // User has disabled local models, so we just make sure other settings are correct.

        if (options.local_files_only) {
            throw Error(
                'Invalid configuration detected: local models are disabled (`env.allowLocalModels=false`) but you have requested to only use local models (`local_files_only=true`).',
            );
        } else if (!env.allowRemoteModels) {
            throw Error(
                'Invalid configuration detected: both local and remote models are disabled. Fix by setting `env.allowLocalModels` or `env.allowRemoteModels` to `true`.',
            );
        }
    }

    dispatchCallback(options.progress_callback, {
        status: 'initiate',
        name: path_or_repo_id,
        file: filename,
    });

    /** @type {import('./cache.js').CacheInterface | null} */
    const cache = await getCache(options?.cache_dir);
    return await loadResourceFile(path_or_repo_id, filename, fatal, options, return_path, cache);
}

/**
 * Fetches a text file from a given path and file name.
 *
 * @param {string} modelPath The path to the directory containing the file.
 * @param {string} fileName The name of the file to fetch.
 * @param {boolean} [fatal=true] Whether to throw an error if the file is not found.
 * @param {PretrainedOptions} [options] An object containing optional parameters.
 * @returns {Promise<string|null>} The text content of the file.
 * @throws Will throw an error if the file is not found and `fatal` is true.
 */
export async function getModelText(modelPath, fileName, fatal = true, options = {}) {
    const buffer = await getModelFile(modelPath, fileName, fatal, options, false);
    if (buffer === null) {
        return null;
    }

    const decoder = new TextDecoder('utf-8');
    return decoder.decode(/** @type {Uint8Array} */ (buffer));
}

/**
 * Fetches a JSON file from a given path and file name.
 *
 * @param {string} modelPath The path to the directory containing the file.
 * @param {string} fileName The name of the file to fetch.
 * @param {boolean} [fatal=true] Whether to throw an error if the file is not found.
 * @param {PretrainedOptions} [options] An object containing optional parameters.
 * @returns {Promise<Object>} The JSON data parsed into a JavaScript object.
 * @throws Will throw an error if the file is not found and `fatal` is true.
 */
export async function getModelJSON(modelPath, fileName, fatal = true, options = {}) {
    const text = await getModelText(modelPath, fileName, fatal, options);
    if (text === null) {
        // Return empty object
        return {};
    }

    return JSON.parse(text);
}
