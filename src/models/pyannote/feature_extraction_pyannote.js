import { FeatureExtractor, validate_audio_inputs } from '../../base/feature_extraction_utils.js';
import { Tensor } from '../../utils/tensor.js';


export class PyAnnoteFeatureExtractor extends FeatureExtractor {
    /**
     * Asynchronously extracts features from a given audio using the provided configuration.
     * @param {Float32Array|Float64Array} audio The audio data as a Float32Array/Float64Array.
     * @returns {Promise<{ input_values: Tensor; }>} The extracted input features.
     */
    async _call(audio) {
        validate_audio_inputs(audio, 'PyAnnoteFeatureExtractor');

        if (audio instanceof Float64Array) {
            audio = new Float32Array(audio);
        }

        const shape = [
            1,            /* batch_size */
            1,            /* num_channels */
            audio.length, /* num_samples */
        ];
        return {
            input_values: new Tensor('float32', audio, shape),
        };
    }

}
