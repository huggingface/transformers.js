import { Tensor } from '../../utils/tensor.js';

/**
 * Compute temporal deltas (and optionally delta-deltas) for [1, T, F] features.
 * @param {Tensor} input_features
 * @param {{order?: 1|2, window?: number, concatenate?: boolean}} [options]
 * @returns {Tensor|{delta: Tensor, delta_delta?: Tensor}}
 */
export function computeTemporalDeltas(input_features, { order = 1, window = 2, concatenate = false } = {}) {
    if (!(input_features instanceof Tensor)) {
        throw new Error('computeTemporalDeltas expects `input_features` as a Tensor.');
    }
    if (input_features.dims.length !== 3 || input_features.dims[0] !== 1) {
        throw new Error(`computeTemporalDeltas expects dims [1, T, F], got [${input_features.dims.join(', ')}].`);
    }
    if (!Number.isInteger(window) || window <= 0) {
        throw new Error('computeTemporalDeltas expects `window` to be a positive integer.');
    }
    if (order !== 1 && order !== 2) {
        throw new Error('computeTemporalDeltas expects `order` to be 1 or 2.');
    }

    const [batch, T, F] = input_features.dims;
    const base = /** @type {Float32Array} */ (input_features.data);
    const delta = new Float32Array(base.length);
    const denom = 2 * Array.from({ length: window }, (_, i) => (i + 1) ** 2).reduce((a, b) => a + b, 0);

    const at = (t, f) => base[t * F + f];
    for (let t = 0; t < T; ++t) {
        for (let f = 0; f < F; ++f) {
            let num = 0;
            for (let n = 1; n <= window; ++n) {
                const tp = Math.min(T - 1, t + n);
                const tm = Math.max(0, t - n);
                num += n * (at(tp, f) - at(tm, f));
            }
            delta[t * F + f] = num / denom;
        }
    }

    const delta_tensor = new Tensor('float32', delta, [batch, T, F]);
    if (order === 1) {
        if (!concatenate) {
            return { delta: delta_tensor };
        }
        return new Tensor('float32', concatFloat32([base, delta]), [batch, T, F * 2]);
    }

    const recursive_result = /** @type {{delta: Tensor}} */ (
        computeTemporalDeltas(delta_tensor, { order: 1, window, concatenate: false })
    );
    const delta_delta_tensor = recursive_result.delta;
    if (!concatenate) {
        return {
            delta: delta_tensor,
            delta_delta: delta_delta_tensor,
        };
    }

    const delta_delta = /** @type {Float32Array} */ (delta_delta_tensor.data);
    return new Tensor('float32', concatFloat32([base, delta, delta_delta]), [batch, T, F * 3]);
}

function concatFloat32(items) {
    const total = items.reduce((sum, arr) => sum + arr.length, 0);
    const output = new Float32Array(total);
    let offset = 0;
    for (const arr of items) {
        output.set(arr, offset);
        offset += arr.length;
    }
    return output;
}
