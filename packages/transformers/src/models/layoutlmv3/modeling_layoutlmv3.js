import { PreTrainedModel } from '../modeling_utils.js';
import { SequenceClassifierOutput, TokenClassifierOutput } from '../modeling_outputs.js';

export class LayoutLMv3PreTrainedModel extends PreTrainedModel {}

/**
 * The base LayoutLMv3 model outputting raw hidden-states without any specific head on top.
 */
export class LayoutLMv3Model extends LayoutLMv3PreTrainedModel {}

/**
 * LayoutLMv3 with a token classification head on top (a linear layer on top of the hidden-states output).
 */
export class LayoutLMv3ForTokenClassification extends LayoutLMv3PreTrainedModel {
    /**
     * Calls the model on new inputs.
     *
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<TokenClassifierOutput>} An object containing the model's output logits for token classification.
     */
    async _call(model_inputs) {
        return new TokenClassifierOutput(await super._call(model_inputs));
    }
}

/**
 * LayoutLMv3 with a sequence classification head on top (a linear layer on top of the pooled output).
 */
export class LayoutLMv3ForSequenceClassification extends LayoutLMv3PreTrainedModel {
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
