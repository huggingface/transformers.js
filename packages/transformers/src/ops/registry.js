const implementations = new Map();
let defaultImplementation = null;

/**
 * Runtime-neutral tensor operation registry. Providers register operations under the same stable
 * backend identifier used by `BackendTensorStorage.backend`.
 */
export class TensorOpRegistry {
    /**
     * Register tensor operations. The one-argument form installs a default implementation and, if
     * `implementation.backend` is present, also registers it for that backend.
     *
     * @param {string|Object} backend
     * @param {Object} [implementation]
     */
    static register(backend, implementation = undefined) {
        if (implementation === undefined) {
            implementation = backend;
            defaultImplementation = implementation;
            if (typeof implementation.backend === 'string') {
                implementations.set(implementation.backend, implementation);
            }
            return;
        }
        if (typeof backend !== 'string' || backend.length === 0) {
            throw new TypeError('Tensor operation backends require a non-empty identifier.');
        }
        implementations.set(backend, implementation);
    }

    /**
     * Remove operations registered for a backend. Intended for provider teardown and tests.
     * @param {string} backend
     * @param {Object} [implementation]
     * @returns {boolean}
     */
    static unregister(backend, implementation = undefined) {
        if (implementation !== undefined && implementations.get(backend) !== implementation) return false;
        return implementations.delete(backend);
    }

    /**
     * Resolve an operation for the backend that owns the input tensors.
     *
     * @param {string} name
     * @param {ReadonlyArray<unknown>} [inputs]
     * @param {{required?: boolean, fallback?: boolean}} [options]
     * @returns {Promise<Function|null>}
     */
    static async resolve(name, inputs = [], { required = true, fallback = true } = {}) {
        const backend = getInputBackend(inputs);
        let implementation = backend ? implementations.get(backend) : null;

        if (!implementation?.[name] && fallback) {
            if (!defaultImplementation) {
                const { getOnnxProviderModule } = await import('../backends/default.js');
                await getOnnxProviderModule();
            }
            implementation = defaultImplementation;
        }

        const operation = await implementation?.[name];
        if (typeof operation === 'function') return operation.bind(implementation);
        if (!required) return null;
        const qualifier = backend ? ` for backend "${backend}"` : '';
        throw new Error(`Tensor operation "${name}"${qualifier} requires a registered implementation.`);
    }

    static get nearest_interpolate_4d() {
        return this.resolve('nearest_interpolate_4d');
    }
    static get bilinear_interpolate_4d() {
        return this.resolve('bilinear_interpolate_4d');
    }
    static get bicubic_interpolate_4d() {
        return this.resolve('bicubic_interpolate_4d');
    }
    static get matmul() {
        return this.resolve('matmul');
    }
    static get stft() {
        return this.resolve('stft');
    }
    static get rfft() {
        return this.resolve('rfft');
    }
    static get top_k() {
        return this.resolve('top_k');
    }
    static get slice() {
        return this.resolve('slice');
    }
}

function getInputBackend(inputs) {
    const backends = new Set();
    for (const input of inputs) {
        const backend = input?.backend ?? input?.getBackendStorage?.()?.backend;
        if (backend && backend !== 'cpu') backends.add(backend);
    }
    if (backends.size > 1) {
        throw new Error(`Tensor operation inputs belong to different backends: ${Array.from(backends).join(', ')}.`);
    }
    return backends.values().next().value ?? null;
}
