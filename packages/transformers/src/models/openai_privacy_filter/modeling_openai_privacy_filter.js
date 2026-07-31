import { PreTrainedModel } from '../modeling_utils.js';
import { SequenceClassifierOutput } from '../modeling_outputs.js';

export class OpenAIPrivacyFilterPreTrainedModel extends PreTrainedModel {}
export class OpenAIPrivacyFilterModel extends OpenAIPrivacyFilterPreTrainedModel {}

export class OpenAIPrivacyFilterForTokenClassification extends OpenAIPrivacyFilterPreTrainedModel {
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
