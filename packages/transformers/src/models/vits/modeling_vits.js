import { PreTrainedModel } from '../modeling_utils.js';
import { ModelOutput } from '../modeling_outputs.js';

/**
 * Describes the outputs for the VITS model.
 */
export class VitsModelOutput extends ModelOutput {
    /**
     * @typedef {import('../../utils/tensor.js').Tensor} Tensor
     */

    /**
     * @param {Object} output The output of the model.
     * @param {Tensor} output.waveform The final audio waveform predicted by the model, of shape `(batch_size, sequence_length)`.
     * @param {Tensor} output.spectrogram The log-mel spectrogram predicted at the output of the flow model.
     * This spectrogram is passed to the Hi-Fi GAN decoder model to obtain the final audio waveform.
     */
    constructor({ waveform, spectrogram }) {
        super();
        this.waveform = waveform;
        this.spectrogram = spectrogram;
    }
}

export class VitsPreTrainedModel extends PreTrainedModel {}

/**
 * The complete VITS model, for text-to-speech synthesis.
 *
 * **Example:** Generate speech from text with `VitsModel`.
 * ```javascript
 * import { AutoTokenizer, VitsModel } from '@huggingface/transformers';
 *
 * // Load the tokenizer and model
 * const tokenizer = await AutoTokenizer.from_pretrained('Xenova/mms-tts-eng');
 * const model = await VitsModel.from_pretrained('Xenova/mms-tts-eng');
 *
 * // Run tokenization
 * const inputs = tokenizer('I love transformers');
 *
 * // Generate waveform
 * const { waveform } = await model(inputs);
 * // Tensor {
 * //   dims: [ 1, 35328 ],
 * //   type: 'float32',
 * //   data: Float32Array(35328) [ ... ],
 * //   size: 35328,
 * // }
 * ```
 */
export class VitsModel extends VitsPreTrainedModel {
    /**
     * Calls the model on new inputs.
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<VitsModelOutput>} The outputs for the VITS model.
     */
    async _call(model_inputs) {
        return new VitsModelOutput(await super._call(model_inputs));
    }
}
