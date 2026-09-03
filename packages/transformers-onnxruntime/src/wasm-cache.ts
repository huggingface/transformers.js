import { getOnnxProviderHost } from './host.js';

async function loadAndCacheFile(url: string): Promise<Response | null | string> {
    const { env, logger, getCache } = getOnnxProviderHost();
    const fileName = url.split('/').pop();
    let cache: any;
    try {
        cache = await getCache?.();
        const cached = await cache?.match(url);
        if (cached) return cached;
    } catch (error) {
        logger.warn(`Failed to load ${fileName} from cache:`, error);
    }

    const response = await env.fetch(url);
    if (!response.ok) throw new Error(`Failed to fetch ${fileName}: ${response.status} ${response.statusText}`);
    if (cache) {
        try {
            await cache.put(url, response.clone());
        } catch (error) {
            logger.warn(`Failed to cache ${fileName}:`, error);
        }
    }
    return response;
}

export async function loadWasmBinary(url: string): Promise<ArrayBuffer | null> {
    const response = await loadAndCacheFile(url);
    if (!response || typeof response === 'string') return null;
    try {
        return await response.arrayBuffer();
    } catch (error) {
        getOnnxProviderHost().logger.warn('Failed to read WASM binary:', error);
        return null;
    }
}

export async function loadWasmFactory(url: string): Promise<string | null> {
    const { apis, logger } = getOnnxProviderHost();
    if (apis.IS_SERVICE_WORKER_ENV || apis.IS_CHROME_AVAILABLE) return url;
    const response = await loadAndCacheFile(url);
    if (!response || typeof response === 'string') return null;
    try {
        const code = (await response.text()).replaceAll('globalThis.process?.versions?.node', 'false');
        return URL.createObjectURL(new Blob([code], { type: 'text/javascript' }));
    } catch (error) {
        logger.warn('Failed to read WASM factory:', error);
        return null;
    }
}
