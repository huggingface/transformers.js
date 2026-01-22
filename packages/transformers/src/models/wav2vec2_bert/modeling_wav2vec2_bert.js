import { PreTrainedModel } from '../modeling_utils.js';
import { CausalLMOutput, SequenceClassifierOutput } from '../modeling_outputs.js';
import { Tensor } from '../../utils/tensor.js';

export class Wav2Vec2BertPreTrainedModel extends PreTrainedModel {}

/**
 * The bare Wav2Vec2Bert Model transformer outputting raw hidden-states without any specific head on top.
 */
export class Wav2Vec2BertModel extends Wav2Vec2BertPreTrainedModel {}

/**
 * Wav2Vec2Bert Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).
 */
export class Wav2Vec2BertForCTC extends Wav2Vec2BertPreTrainedModel {
    /**
     * @param {Object} model_inputs
     * @param {Tensor} model_inputs.input_features Float values of input mel-spectrogram.
     * @param {Tensor} model_inputs.attention_mask Mask to avoid performing convolution and attention on padding token indices. Mask values selected in [0, 1]
     */
    async _call(model_inputs) {
        return new CausalLMOutput(await super._call(model_inputs));
    }
}

/**
 * Wav2Vec2Bert Model with a sequence classification head on top (a linear layer over the pooled output).
 */
export class Wav2Vec2BertForSequenceClassification extends Wav2Vec2BertPreTrainedModel {
    /**
     * Calls the model on new inputs.
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<SequenceClassifierOutput>} An object containing the model's output logits for sequence classification.
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
