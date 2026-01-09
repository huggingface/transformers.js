import { PreTrainedModel } from '../_base/pre-trained-model.js';
import { CausalLMOutput } from '../_base/output.js';
import { Tensor } from '../../utils/tensor.js';

export class ParakeetPreTrainedModel extends PreTrainedModel {}
export class ParakeetForCTC extends ParakeetPreTrainedModel {
    /**
     * @param {Object} model_inputs
     * @param {Tensor} model_inputs.input_values Float values of input raw speech waveform.
     * @param {Tensor} model_inputs.attention_mask Mask to avoid performing convolution and attention on padding token indices. Mask values selected in [0, 1]
     */
    async _call(model_inputs) {
        return new CausalLMOutput(await super._call(model_inputs));
    }
}
