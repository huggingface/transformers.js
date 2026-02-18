import { PreTrainedModel } from '../modeling_utils.js';
import {
    MaskedLMOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
} from '../modeling_outputs.js';

export class MPNetPreTrainedModel extends PreTrainedModel {}

/**
 * The bare MPNet Model transformer outputting raw hidden-states without any specific head on top.
 */
export class MPNetModel extends MPNetPreTrainedModel {}

/**
 * MPNetForMaskedLM is a class representing a MPNet model for masked language modeling.
 */
export class MPNetForMaskedLM extends MPNetPreTrainedModel {
    /**
     * Calls the model on new inputs.
     *
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<MaskedLMOutput>} An object containing the model's output logits for masked language modeling.
     */
    async _call(model_inputs) {
        return new MaskedLMOutput(await super._call(model_inputs));
    }
}

/**
 * MPNetForSequenceClassification is a class representing a MPNet model for sequence classification.
 */
export class MPNetForSequenceClassification extends MPNetPreTrainedModel {
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

/**
 * MPNetForTokenClassification is a class representing a MPNet model for token classification.
 */
export class MPNetForTokenClassification extends MPNetPreTrainedModel {
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
 * MPNetForQuestionAnswering is a class representing a MPNet model for question answering.
 */
export class MPNetForQuestionAnswering extends MPNetPreTrainedModel {
    /**
     * Calls the model on new inputs.
     *
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<QuestionAnsweringModelOutput>} An object containing the model's output logits for question answering.
     */
    async _call(model_inputs) {
        return new QuestionAnsweringModelOutput(await super._call(model_inputs));
    }
}
