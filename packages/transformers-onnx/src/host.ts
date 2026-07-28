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

export interface OnnxProviderHost {
    readonly env: any;
    readonly apis: any;
    readonly logger: any;
    getModelFile(modelId: string, file: string, fatal: boolean, options: any, returnPath?: boolean): Promise<any>;
    getCacheNames(config: any, options: any): Set<string>;
    createBackendTensor(storage: BackendTensorStorage): any;
    getBackendTensorStorage(tensor: any): BackendTensorStorage | null;
    getCache?(): Promise<any>;
}

let configuredHost: OnnxProviderHost | null = null;

const fallbackEnvironment: any = {
    backends: { onnx: {} },
    logLevel: 30,
    useWasmCache: typeof caches !== 'undefined',
    fetch: (...args: any[]) => (globalThis.fetch as any)(...args),
};

const environment = new Proxy(fallbackEnvironment, {
    get(target, property) {
        return (configuredHost?.env ?? target)[property];
    },
    set(target, property, value) {
        (configuredHost?.env ?? target)[property] = value;
        return true;
    },
});

const apis = {
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

const logger = new Proxy(console, {
    get(target, property) {
        return (configuredHost?.logger ?? target)[property];
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
};

export function configureOnnxProviderHost(host: OnnxProviderHost): void {
    if (fallbackEnvironment.backends.onnx) {
        host.env.backends.onnx = fallbackEnvironment.backends.onnx;
    }
    configuredHost = host;
}

export function getOnnxProviderHost(): OnnxProviderHost {
    return configuredHost ?? fallbackHost;
}
