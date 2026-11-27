import { FeatureExtractor, validate_audio_inputs } from '../../feature_extraction_utils.js';
import { Tensor, full, matmul, add, mul, pow } from '../../utils/tensor.js';

export class Wav2Vec2FeatureExtractor extends FeatureExtractor {
    /**
     * @param {Float32Array} input_values
     * @returns {Promise<Float32Array>}
     */
    async _zero_mean_unit_var_norm(input_values) {
        // TODO support batch?
        const length = input_values.length;
        if (length === 0) return input_values;

        // (x - mean) / sqrt(var + 1e-7), with means computed as [1, length] @ [length, 1] products
        const x = new Tensor('float32', input_values, [length, 1]);
        const centered = await add(x, await matmul(full([1, length], -1 / length), x));
        const variance = await matmul(full([1, length], 1 / length), await pow(centered, 2));
        const normalized = await mul(centered, await pow(await add(variance, 1e-7), -0.5));
        return /** @type {Float32Array} */ (normalized.data);
    }

    /**
     * Asynchronously extracts features from a given audio using the provided configuration.
     * @param {Float32Array|Float64Array} audio The audio data as a Float32Array/Float64Array.
     * @returns {Promise<{ input_values: Tensor; attention_mask: Tensor }>} A Promise resolving to an object containing the extracted input features and attention mask as Tensors.
     */
    async _call(audio) {
        validate_audio_inputs(audio, 'Wav2Vec2FeatureExtractor');

        if (audio instanceof Float64Array) {
            audio = new Float32Array(audio);
        }

        let input_values = audio;

        // zero-mean and unit-variance normalization
        if (this.config.do_normalize) {
            input_values = await this._zero_mean_unit_var_norm(input_values);
        }

        // TODO: allow user to pass in attention mask
        const shape = [1, input_values.length];
        return {
            input_values: new Tensor('float32', input_values, shape),
            attention_mask: new Tensor('int64', new BigInt64Array(input_values.length).fill(1n), shape),
        };
    }
}
