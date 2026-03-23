import { PreTrainedModel } from '../modeling_utils.js';
import { MaskedLMOutput, QuestionAnsweringModelOutput, SequenceClassifierOutput } from '../modeling_outputs.js';

export class AlbertPreTrainedModel extends PreTrainedModel {}
export class AlbertModel extends AlbertPreTrainedModel {}
export class AlbertForSequenceClassification extends AlbertPreTrainedModel {
    /**
     * Calls the model on new inputs.
     *
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<SequenceClassifierOutput>} returned object
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
export class AlbertForQuestionAnswering extends AlbertPreTrainedModel {
    /**
     * Calls the model on new inputs.
     *
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<QuestionAnsweringModelOutput>} returned object
     */
    async _call(model_inputs) {
        return new QuestionAnsweringModelOutput(await super._call(model_inputs));
    }
}
export class AlbertForMaskedLM extends AlbertPreTrainedModel {
    /**
     * Calls the model on new inputs.
     *
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<MaskedLMOutput>} returned object
     */
    async _call(model_inputs) {
        return new MaskedLMOutput(await super._call(model_inputs));
    }
}
