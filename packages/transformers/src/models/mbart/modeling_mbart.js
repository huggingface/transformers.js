import { PreTrainedModel } from '../modeling_utils.js';
import { SequenceClassifierOutput } from '../modeling_outputs.js';

export class MBartPreTrainedModel extends PreTrainedModel {}

/**
 * The bare MBART Model outputting raw hidden-states without any specific head on top.
 */
export class MBartModel extends MBartPreTrainedModel {}

/**
 * The MBART Model with a language modeling head. Can be used for summarization, after fine-tuning the pretrained models.
 */
export class MBartForConditionalGeneration extends MBartPreTrainedModel {}

/**
 * MBart model with a sequence classification/head on top (a linear layer on top of the pooled output).
 */
export class MBartForSequenceClassification extends MBartPreTrainedModel {
    /**
     * Calls the model on new inputs.
     *
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<SequenceClassifierOutput>} An object containing the model's output logits for sequence classification.
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}

export class MBartForCausalLM extends MBartPreTrainedModel {}
