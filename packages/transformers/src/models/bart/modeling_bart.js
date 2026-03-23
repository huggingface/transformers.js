import { PreTrainedModel } from '../modeling_utils.js';
import { SequenceClassifierOutput } from '../modeling_outputs.js';

export class BartPretrainedModel extends PreTrainedModel {}

/**
 * The bare BART Model outputting raw hidden-states without any specific head on top.
 */
export class BartModel extends BartPretrainedModel {}

/**
 * The BART Model with a language modeling head. Can be used for summarization.
 */
export class BartForConditionalGeneration extends BartPretrainedModel {}

/**
 * Bart model with a sequence classification/head on top (a linear layer on top of the pooled output)
 */
export class BartForSequenceClassification extends BartPretrainedModel {
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
