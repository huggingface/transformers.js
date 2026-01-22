import { PreTrainedModel } from '../modeling_utils.js';

export class MoonshinePreTrainedModel extends PreTrainedModel {
    requires_attention_mask = false;
    main_input_name = 'input_values';
    forward_params = ['input_values', 'decoder_input_ids', 'past_key_values'];
}

/**
 * MoonshineModel class for training Moonshine models without a language model head.
 */
export class MoonshineModel extends MoonshinePreTrainedModel {}

export class MoonshineForConditionalGeneration extends MoonshinePreTrainedModel {}
