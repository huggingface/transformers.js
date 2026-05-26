import { Qwen2VLForConditionalGeneration, Qwen2VLForCausalLM } from '../qwen2_vl/modeling_qwen2_vl.js';

export class Qwen2_5_VLForConditionalGeneration extends Qwen2VLForConditionalGeneration {
    /** @override */
    image_grid_thw_name = 'image_grid_thw';
}

export class Qwen2_5_VLForCausalLM extends Qwen2VLForCausalLM {
    /** @override */
    image_grid_thw_name = 'image_grid_thw';
}
