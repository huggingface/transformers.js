import { PreTrainedModel } from '../modeling_utils.js';
import {
    MaskedLMOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
} from '../modeling_outputs.js';

export class RobertaPreTrainedModel extends PreTrainedModel {}
export class RobertaModel extends RobertaPreTrainedModel {}

/**
 * RobertaForMaskedLM class for performing masked language modeling on Roberta models.
 */
export class RobertaForMaskedLM extends RobertaPreTrainedModel {
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

/**
 * RobertaForSequenceClassification class for performing sequence classification on Roberta models.
 */
export class RobertaForSequenceClassification extends RobertaPreTrainedModel {
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

/**
 * RobertaForTokenClassification class for performing token classification on Roberta models.
 */
export class RobertaForTokenClassification extends RobertaPreTrainedModel {
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
 * RobertaForQuestionAnswering class for performing question answering on Roberta models.
 */
export class RobertaForQuestionAnswering extends RobertaPreTrainedModel {
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
