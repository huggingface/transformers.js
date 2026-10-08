import { env } from '../../env.js';
import { getCacheForEnvironment, tryCache } from '../cache.js';
import { FileCache } from '../cache/FileCache.js';
import { FileResponse } from './FileResponse.js';
import { isValidHfModelId, isValidUrl, makePretrainedOptionsKey, pathJoin } from './utils.js';

// Only assets explicitly listed here receive new cache keys. Model weights and
// all other resources retain their existing keys. Versions affect storage, not
// source URLs or the contents of downloaded files.
const ASSET_CACHE_VERSIONS = new Map([['tokenizer_config.json', 2]]);

const OBJECT_IDS = new WeakMap();
let nextObjectId = 0;

/** @param {object|null|undefined} object @private */
function objectId(object) {
    if (!object) return null;
    if (!OBJECT_IDS.has(object)) OBJECT_IDS.set(object, ++nextObjectId);
    return OBJECT_IDS.get(object);
}

/**
 * Encode a version in the backend's existing key format.
 * @private
 * @param {string} key
 * @param {number} version
 * @param {import('../cache.js').CacheInterface|null} cache
 */
function versionedKey(key, version, cache) {
    if (!version) return key;
    if (cache instanceof FileCache) {
        return pathJoin(`transformersjs_assets_v${version}`, key);
    }
    // Preserve relative URLs as well as existing query parameters and fragments.
    const hashIndex = key.indexOf('#');
    const url = hashIndex < 0 ? key : key.slice(0, hashIndex);
    const hash = hashIndex < 0 ? '' : key.slice(hashIndex);
    return `${url}${url.includes('?') ? '&' : '?'}__transformersjs_cache_version=${version}${hash}`;
}

/**
 * Keep the legacy cache-lookup helper consistent with versioned resolution.
 * @private
 * @param {import('../cache.js').CacheInterface|null} cache
 * @param {string} localPath
 * @param {string} proposedCacheKey
 * @returns {string[]}
 */
export function getCacheLookupKeys(cache, localPath, proposedCacheKey) {
    const version =
        cache instanceof FileCache
            ? Number(proposedCacheKey.match(/^transformersjs_assets_v(\d+)\//)?.[1] ?? 0)
            : Number(proposedCacheKey.match(/[?&]__transformersjs_cache_version=(\d+)(?:[&#]|$)/)?.[1] ?? 0);
    return [...new Set([versionedKey(localPath, version, cache), proposedCacheKey])];
}

/**
 * Internal resource-loading dependency. Captures only source, transport, and
 * cache settings at the loading boundary; public pretrained options stay intact.
 * @private
 */
export class AssetLoadingContext {
    /** @param {import('../hub.js').PretrainedOptions} [options] */
    constructor(options = {}) {
        const {
            version,
            allowLocalModels,
            allowRemoteModels,
            localModelPath,
            remoteHost,
            remotePathTemplate,
            useFS,
            fetch,
            useCustomCache,
            customCache,
            experimental_useCrossOriginStorage,
            useBrowserCache,
            cacheKey,
            useFSCache,
            cacheDir,
        } = env;
        this.settings = {
            version,
            allowLocalModels,
            allowRemoteModels,
            localModelPath,
            remoteHost,
            remotePathTemplate,
            useFS,
            fetch,
            useCustomCache,
            customCache,
            experimental_useCrossOriginStorage,
            useBrowserCache,
            cacheKey,
            useFSCache,
            cacheDir,
        };
        this.options = {
            revision: options.revision ?? 'main',
            cache_dir: options.cache_dir ?? null,
            local_files_only: options.local_files_only ?? false,
        };
        const isNode = typeof process !== 'undefined' && process?.release?.name === 'node';
        this.userAgent = isNode ? `transformers.js/${version}; is_ci/${!!process.env?.TESTING_REMOTELY};` : null;
        this.token = isNode ? (process.env?.HF_TOKEN ?? process.env?.HF_ACCESS_TOKEN) : null;
        /** @type {Promise<import('../cache.js').CacheInterface|null>|undefined} */
        this.cache = undefined;
    }

    getCache() {
        return (this.cache ??= getCacheForEnvironment(this.settings, this.options.cache_dir));
    }

    /** @param {URL|string} urlOrPath */
    getHeaders(urlOrPath) {
        const headers = new Headers();
        if (this.userAgent) headers.set('User-Agent', this.userAgent);
        if (this.token && isValidUrl(urlOrPath, ['http:', 'https:'], ['huggingface.co', 'hf.co'])) {
            headers.set('Authorization', `Bearer ${this.token}`);
        }
        return headers;
    }

    /**
     * @param {URL|string} urlOrPath
     * @param {RequestInit} [init]
     * @returns {Promise<FileResponse|Response>}
     */
    async read(urlOrPath, init = {}) {
        if (this.settings.useFS && !isValidUrl(urlOrPath, ['http:', 'https:', 'blob:'])) {
            return new FileResponse(
                urlOrPath instanceof URL
                    ? urlOrPath.protocol === 'file:'
                        ? urlOrPath.pathname
                        : urlOrPath.toString()
                    : urlOrPath,
            );
        }
        const headers = this.getHeaders(urlOrPath);
        new Headers(init.headers).forEach((value, key) => headers.set(key, value));
        return this.settings.fetch(urlOrPath, { ...init, headers });
    }

    /**
     * @param {string} path_or_repo_id
     * @param {string} filename
     * @param {import('../cache.js').CacheInterface|null} [cache]
     */
    resolve(path_or_repo_id, filename, cache = null) {
        const { revision } = this.options;
        const requestURL = pathJoin(path_or_repo_id, filename);
        const validModelId = isValidHfModelId(path_or_repo_id);
        const localPath = validModelId ? pathJoin(this.settings.localModelPath, requestURL) : requestURL;
        const remoteURL = pathJoin(
            this.settings.remoteHost,
            this.settings.remotePathTemplate
                .replaceAll('{model}', path_or_repo_id)
                .replaceAll('{revision}', encodeURIComponent(revision)),
            filename,
        );
        const baseCacheKey =
            cache instanceof FileCache
                ? revision === 'main'
                    ? requestURL
                    : pathJoin(path_or_repo_id, revision, filename)
                : remoteURL;
        const version = ASSET_CACHE_VERSIONS.get(filename) ?? 0;
        const localCacheKey = versionedKey(localPath, version, cache);
        const proposedCacheKey = versionedKey(baseCacheKey, version, cache);
        const cacheKeys = getCacheLookupKeys(cache, localPath, proposedCacheKey);
        const obsoleteCacheKeys = [];
        for (let previous = 0; previous < version; ++previous) {
            obsoleteCacheKeys.push(
                versionedKey(localPath, previous, cache),
                versionedKey(baseCacheKey, previous, cache),
            );
        }
        return {
            requestURL,
            localPath,
            remoteURL,
            validModelId,
            cacheVersion: version,
            localCacheKey,
            proposedCacheKey,
            cacheKeys,
            obsoleteCacheKeys: [...new Set(obsoleteCacheKeys)],
        };
    }

    /**
     * Return the matching key too: a local cache hit must not be mistaken for a
     * remote-key hit when returning a filesystem path or removing an entry.
     * @param {import('../cache.js').CacheInterface|null} cache
     * @param {ReturnType<AssetLoadingContext['resolve']>} resource
     */
    async match(cache, resource) {
        if (cache) {
            for (const key of resource.cacheKeys) {
                const response = await tryCache(cache, key);
                if (response !== undefined) return { key, response };
            }
        }
        return undefined;
    }

    /**
     * Best-effort cleanup only after a replacement was successfully stored.
     * Custom caches need not implement deletion or enumerate their entries.
     * @param {import('../cache.js').CacheInterface} cache
     * @param {ReturnType<AssetLoadingContext['resolve']>} resource
     * @param {string} storedKey
     */
    async cleanup(cache, resource, storedKey) {
        if (!cache.delete || !resource.obsoleteCacheKeys.length || !(await tryCache(cache, storedKey))) return;
        for (const key of resource.obsoleteCacheKeys) {
            try {
                await cache.delete(key);
            } catch {
                // Cleanup must not turn a successful asset load into a failure.
            }
        }
    }

    /** @param {string} modelId @param {...unknown} parts */
    key(modelId, ...parts) {
        const { fetch, customCache, ...settings } = this.settings;
        return makePretrainedOptionsKey(
            modelId,
            this.options,
            settings,
            objectId(fetch),
            objectId(customCache),
            this.token,
            ...parts,
        );
    }
}
