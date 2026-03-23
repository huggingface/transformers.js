import { PreTrainedModel } from '../modeling_utils.js';
import { CausalLMOutput, SequenceClassifierOutput, TokenClassifierOutput } from '../modeling_outputs.js';
import { Tensor } from '../../utils/tensor.js';

export class UniSpeechSatPreTrainedModel extends PreTrainedModel {}

/**
 * The bare UniSpeechSat Model transformer outputting raw hidden-states without any specific head on top.
 */
export class UniSpeechSatModel extends UniSpeechSatPreTrainedModel {}

/**
 * UniSpeechSat Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).
 */
export class UniSpeechSatForCTC extends UniSpeechSatPreTrainedModel {
    /**
     * @param {Object} model_inputs
     * @param {Tensor} model_inputs.input_values Float values of input raw speech waveform.
     * @param {Tensor} model_inputs.attention_mask Mask to avoid performing convolution and attention on padding token indices. Mask values selected in [0, 1]
     */
    async _call(model_inputs) {
        return new CausalLMOutput(await super._call(model_inputs));
    }
}

/**
 * UniSpeechSat Model with a sequence classification head on top (a linear layer over the pooled output).
 */
export class UniSpeechSatForSequenceClassification extends UniSpeechSatPreTrainedModel {
    /**
     * Calls the model on new inputs.
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<SequenceClassifierOutput>} An object containing the model's output logits for sequence classification.
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}

/**
 * UniSpeechSat Model with a frame classification head on top for tasks like Speaker Diarization.
 */
export class UniSpeechSatForAudioFrameClassification extends UniSpeechSatPreTrainedModel {
    /**
     * Calls the model on new inputs.
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<TokenClassifierOutput>} An object containing the model's output logits for sequence classification.
     */
    async _call(model_inputs) {
        return new TokenClassifierOutput(await super._call(model_inputs));
    }
}
