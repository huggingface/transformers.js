import { LlavaForConditionalGeneration } from '../llava/modeling_llava.js';

export class Lfm3VlForConditionalGeneration extends LlavaForConditionalGeneration {
    forward_params = [
        'input_ids',
        'attention_mask',
        'pixel_values',
        'image_grid_thw',
        'position_ids',
        'past_key_values',
    ];
}
