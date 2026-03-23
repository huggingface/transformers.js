import { PreTrainedModel } from '../modeling_utils.js';

export class T5PreTrainedModel extends PreTrainedModel {
    forward_params = [
        'input_ids',
        'attention_mask',
        'encoder_outputs',
        'decoder_input_ids',
        'decoder_attention_mask',
        'past_key_values',
    ];
}

export class T5Model extends T5PreTrainedModel {}

/**
 * T5Model is a class representing a T5 model for conditional generation.
 */
export class T5ForConditionalGeneration extends T5PreTrainedModel {}
