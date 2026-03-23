import { PreTrainedModel } from '../modeling_utils.js';
import {
    MaskedLMOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
} from '../modeling_outputs.js';

export class ConvBertPreTrainedModel extends PreTrainedModel {}

/**
 * The bare ConvBERT Model transformer outputting raw hidden-states without any specific head on top.
 */
export class ConvBertModel extends ConvBertPreTrainedModel {}

/**
 * ConvBERT Model with a language modeling head on top.
 */
export class ConvBertForMaskedLM extends ConvBertPreTrainedModel {
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
 * ConvBERT Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled output)
 */
export class ConvBertForSequenceClassification extends ConvBertPreTrainedModel {
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
 * ConvBERT Model with a token classification head on top (a linear layer on top of the hidden-states output)
 * e.g. for Named-Entity-Recognition (NER) tasks.
 */
export class ConvBertForTokenClassification extends ConvBertPreTrainedModel {
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
 * ConvBERT Model with a span classification head on top for extractive question-answering tasks like SQuAD
 * (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`)
 */
export class ConvBertForQuestionAnswering extends ConvBertPreTrainedModel {
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
