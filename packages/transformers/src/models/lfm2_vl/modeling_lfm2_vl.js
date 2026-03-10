import { LlavaForConditionalGeneration } from '../llava/modeling_llava.js';

export class Lfm2VlForConditionalGeneration extends LlavaForConditionalGeneration {
    forward_params = [
        'input_ids',
        'attention_mask',
        'pixel_values',
        'pixel_attention_mask',
        'spatial_shapes',
        'position_ids',
        'past_key_values',
    ];
}
