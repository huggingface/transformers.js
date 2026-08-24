export interface BackendTensorStorage {
    readonly backend: string;
    readonly handle: unknown;
    readonly type: string;
    dims: number[];
    readonly data: unknown;
    readonly size: number;
    readonly location: string;
    dispose(): void;
}

export interface OnnxProviderEnvironment {
    backends: Record<string, unknown>;
    logLevel?: number;
    useWasmCache?: boolean;
    fetch: typeof globalThis.fetch;
}

export interface OnnxProviderApis {
    readonly IS_NODE_ENV: boolean;
    readonly IS_WEB_ENV: boolean;
    readonly IS_WEBGPU_AVAILABLE: boolean;
    readonly IS_WEBNN_AVAILABLE: boolean;
    readonly IS_DENO_WEB_RUNTIME: boolean;
    readonly IS_SAFARI_BELOW_26: boolean;
    readonly IS_SERVICE_WORKER_ENV: boolean;
    readonly IS_CHROME_AVAILABLE: boolean;
}

export interface OnnxProviderLogger {
    info(...data: unknown[]): void;
    warn(...data: unknown[]): void;
    error(...data: unknown[]): void;
}

export interface OnnxProviderCache {
    match(request: string): Promise<Response | string | { arrayBuffer(): Promise<ArrayBuffer> } | undefined>;
    put(request: string, response: Response): Promise<void>;
}

export interface OnnxProviderHost {
    readonly env: OnnxProviderEnvironment;
    readonly apis: OnnxProviderApis;
    readonly logger: OnnxProviderLogger;
    getModelFile(
        modelId: string,
        file: string,
        fatal: boolean,
        options: Record<string, unknown>,
        returnPath?: boolean,
    ): Promise<string | Uint8Array>;
    getCacheNames(config: unknown, options: unknown): Set<string>;
    createBackendTensor(storage: BackendTensorStorage): unknown;
    getBackendTensorStorage(tensor: unknown): BackendTensorStorage | null;
    getCache?(): Promise<OnnxProviderCache | null>;
    readonly maxExternalDataChunks: number;
}

const ONNX_HOST_SYMBOL = Symbol.for('transformers.js.onnxProviderHost');
let configuredHost: OnnxProviderHost | null =
    ((globalThis as any)[ONNX_HOST_SYMBOL] as OnnxProviderHost | undefined) ?? null;

const fallbackEnvironment: OnnxProviderEnvironment = {
    backends: { onnx: {} },
    logLevel: 30,
    useWasmCache: typeof caches !== 'undefined',
    fetch: (...args) => globalThis.fetch(...args),
};

const environment = new Proxy(fallbackEnvironment, {
    get(target, property) {
        return Reflect.get(configuredHost?.env ?? target, property);
    },
    set(target, property, value) {
        return Reflect.set(configuredHost?.env ?? target, property, value);
    },
});

const fallbackApis = {
    IS_NODE_ENV: typeof process !== 'undefined' && process?.release?.name === 'node',
    IS_WEB_ENV: typeof window !== 'undefined' || typeof self !== 'undefined',
    IS_WEBGPU_AVAILABLE: typeof navigator !== 'undefined' && !!navigator.gpu,
    IS_WEBNN_AVAILABLE: typeof navigator !== 'undefined' && 'ml' in navigator,
    IS_DENO_WEB_RUNTIME: 'Deno' in globalThis && typeof window !== 'undefined',
    IS_SAFARI_BELOW_26: false,
    IS_SERVICE_WORKER_ENV:
        'ServiceWorkerGlobalScope' in globalThis && globalThis instanceof (globalThis as any).ServiceWorkerGlobalScope,
    IS_CHROME_AVAILABLE: 'chrome' in globalThis,
};

const apis = new Proxy(fallbackApis, {
    get(target, property) {
        return Reflect.get(configuredHost?.apis ?? target, property);
    },
});

const logger = new Proxy(console, {
    get(target, property) {
        return Reflect.get(configuredHost?.logger ?? target, property);
    },
});

const fallbackHost: OnnxProviderHost = {
    env: environment,
    apis,
    logger,
    async getModelFile() {
        throw new Error('OnnxInferenceProvider host does not provide model file loading.');
    },
    getCacheNames() {
        return new Set();
    },
    createBackendTensor() {
        throw new Error('OnnxInferenceProvider host does not provide tensor creation.');
    },
    getBackendTensorStorage() {
        return null;
    },
    maxExternalDataChunks: 100,
};

export function configureOnnxProviderHost(host: OnnxProviderHost): void {
    // Direct package imports initialize ORT against the fallback environment, so migrate those
    // settings when a host arrives later. A symbol-registered host was already configured before
    // module evaluation and contains the authoritative ORT environment.
    if (configuredHost === null && fallbackEnvironment.backends.onnx) {
        const target = (host.env.backends.onnx ?? {}) as Record<string, any>;
        const source = fallbackEnvironment.backends.onnx as Record<string, any>;
        const targetWasm = target.wasm;
        const targetWebgpu = target.webgpu;
        Object.assign(target, source);
        target.wasm = Object.assign(source.wasm ?? {}, targetWasm ?? {});
        target.webgpu = Object.assign(source.webgpu ?? {}, targetWebgpu ?? {});
        host.env.backends.onnx = target;
    }
    configuredHost = host;
}

export function getOnnxProviderHost(): OnnxProviderHost {
    return configuredHost ?? fallbackHost;
}
