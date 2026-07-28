let implementation = null;

/**
 * Runtime-neutral tensor operation registry. Inference providers install optimized implementations.
 */
export class TensorOpRegistry {
    static register(provider) {
        implementation = provider;
    }

    static get nearest_interpolate_4d() {
        return getOperation('nearest_interpolate_4d');
    }
    static get bilinear_interpolate_4d() {
        return getOperation('bilinear_interpolate_4d');
    }
    static get bicubic_interpolate_4d() {
        return getOperation('bicubic_interpolate_4d');
    }
    static get matmul() {
        return getOperation('matmul');
    }
    static get stft() {
        return getOperation('stft');
    }
    static get rfft() {
        return getOperation('rfft');
    }
    static get top_k() {
        return getOperation('top_k');
    }
    static get slice() {
        return getOperation('slice');
    }
}

async function getOperation(name) {
    if (!implementation) {
        await import('../backends/default.js');
    }
    if (!implementation) {
        throw new Error(`Tensor operation "${name}" requires an installed inference provider.`);
    }
    return implementation[name];
}
