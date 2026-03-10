import { LlavaForConditionalGeneration } from '../llava/modeling_llava.js';

/**
 * The Idefics3 model which consists of a vision backbone and a language model.
 */
export class Idefics3ForConditionalGeneration extends LlavaForConditionalGeneration {
    forward_params = [
        'input_ids',
        'attention_mask',
        'pixel_values',
        'pixel_attention_mask',
        'position_ids',
        'past_key_values',
    ];
}
