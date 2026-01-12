import { PreTrainedModel } from '../modeling_utils.js';
import {
    MaskedLMOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
} from '../modeling_outputs.js';

export class XLMPreTrainedModel extends PreTrainedModel {}

/**
 * The bare XLM Model transformer outputting raw hidden-states without any specific head on top.
 */
export class XLMModel extends XLMPreTrainedModel {}

/**
 * The XLM Model transformer with a language modeling head on top (linear layer with weights tied to the input embeddings).
 */
export class XLMWithLMHeadModel extends XLMPreTrainedModel {
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
 * XLM Model with a sequence classification/regression head on top (a linear layer on top of the pooled output)
 */
export class XLMForSequenceClassification extends XLMPreTrainedModel {
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
 * XLM Model with a token classification head on top (a linear layer on top of the hidden-states output)
 */
export class XLMForTokenClassification extends XLMPreTrainedModel {
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
 * XLM Model with a span classification head on top for extractive question-answering tasks
 */
export class XLMForQuestionAnswering extends XLMPreTrainedModel {
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
