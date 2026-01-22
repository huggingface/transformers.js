/**
 * @typedef {import('../utils/tensor.js').Tensor} Tensor
 */

export class ModelOutput {}

/**
 * Base class for model's outputs, with potential hidden states and attentions.
 */
export class BaseModelOutput extends ModelOutput {
    /**
     * @param {Object} output The output of the model.
     * @param {Tensor} output.last_hidden_state Sequence of hidden-states at the output of the last layer of the model.
     * @param {Tensor} [output.hidden_states] Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
     * @param {Tensor} [output.attentions] Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
     */
    constructor({ last_hidden_state, hidden_states = null, attentions = null }) {
        super();
        this.last_hidden_state = last_hidden_state;
        this.hidden_states = hidden_states;
        this.attentions = attentions;
    }
}

/**
 * Base class for outputs of sentence classification models.
 */
export class SequenceClassifierOutput extends ModelOutput {
    /**
     * @param {Object} output The output of the model.
     * @param {Tensor} output.logits classification (or regression if config.num_labels==1) scores (before SoftMax).
     * @param {Record<string, Tensor>} [output.attentions] Object of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
     * Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
     */
    constructor({ logits, ...attentions }) {
        super();
        this.logits = logits;
        const attentions_list = Object.values(attentions);
        if (attentions_list.length > 0) {
            // Only set attentions if they are not empty
            this.attentions = attentions_list;
        }
    }
}

/**
 * Base class for outputs of token classification models.
 */
export class TokenClassifierOutput extends ModelOutput {
    /**
     * @param {Object} output The output of the model.
     * @param {Tensor} output.logits Classification scores (before SoftMax).
     */
    constructor({ logits }) {
        super();
        this.logits = logits;
    }
}

/**
 * Base class for masked language models outputs.
 */
export class MaskedLMOutput extends ModelOutput {
    /**
     * @param {Object} output The output of the model.
     * @param {Tensor} output.logits Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
     */
    constructor({ logits }) {
        super();
        this.logits = logits;
    }
}

/**
 * Base class for outputs of question answering models.
 */
export class QuestionAnsweringModelOutput extends ModelOutput {
    /**
     * @param {Object} output The output of the model.
     * @param {Tensor} output.start_logits Span-start scores (before SoftMax).
     * @param {Tensor} output.end_logits Span-end scores (before SoftMax).
     */
    constructor({ start_logits, end_logits }) {
        super();
        this.start_logits = start_logits;
        this.end_logits = end_logits;
    }
}

/**
 * Base class for causal language model (or autoregressive) outputs.
 */
export class CausalLMOutput extends ModelOutput {
    /**
     * @param {Object} output The output of the model.
     * @param {Tensor} output.logits Prediction scores of the language modeling head (scores for each vocabulary token before softmax).
     */
    constructor({ logits }) {
        super();
        this.logits = logits;
    }
}

/**
 * Base class for causal language model (or autoregressive) outputs.
 */
export class CausalLMOutputWithPast extends ModelOutput {
    /**
     * @param {Object} output The output of the model.
     * @param {Tensor} output.logits Prediction scores of the language modeling head (scores for each vocabulary token before softmax).
     * @param {Tensor} output.past_key_values Contains pre-computed hidden-states (key and values in the self-attention blocks)
     * that can be used (see `past_key_values` input) to speed up sequential decoding.
     */
    constructor({ logits, past_key_values }) {
        super();
        this.logits = logits;
        this.past_key_values = past_key_values;
    }
}

export class Seq2SeqLMOutput extends ModelOutput {
    /**
     * @param {Object} output The output of the model.
     * @param {Tensor} output.logits The output logits of the model.
     * @param {Tensor} output.past_key_values An tensor of key/value pairs that represent the previous state of the model.
     * @param {Tensor} output.encoder_outputs The output of the encoder in a sequence-to-sequence model.
     * @param {Tensor} [output.decoder_attentions] Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the self-attention heads.
     * @param {Tensor} [output.cross_attentions] Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the weighted average in the cross-attention heads.
     */
    constructor({ logits, past_key_values, encoder_outputs, decoder_attentions = null, cross_attentions = null }) {
        super();
        this.logits = logits;
        this.past_key_values = past_key_values;
        this.encoder_outputs = encoder_outputs;
        this.decoder_attentions = decoder_attentions;
        this.cross_attentions = cross_attentions;
    }
}

export class ImageMattingOutput extends ModelOutput {
    /**
     * @param {Object} output The output of the model.
     * @param {Tensor} output.alphas Estimated alpha values, of shape `(batch_size, num_channels, height, width)`.
     */
    constructor({ alphas }) {
        super();
        this.alphas = alphas;
    }
}
