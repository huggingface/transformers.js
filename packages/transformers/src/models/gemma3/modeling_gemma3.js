import { PreTrainedModel } from '../modeling_utils.js';
import { LlavaForConditionalGeneration } from '../llava/modeling_llava.js';

/**
 * The bare Gemma3 Model outputting raw hidden-states without any specific head on top.
 */
export class Gemma3PreTrainedModel extends PreTrainedModel {}

/**
 * The bare Gemma3 Model outputting raw hidden-states without any specific head on top.
 */
export class Gemma3Model extends Gemma3PreTrainedModel {}

export class Gemma3ForConditionalGeneration extends LlavaForConditionalGeneration {}

export class Gemma3ForCausalLM extends Gemma3ForConditionalGeneration {}
