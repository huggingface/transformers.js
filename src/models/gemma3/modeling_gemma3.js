import { PreTrainedModel } from '../_base/pre-trained-model.js';

/**
 * The bare Gemma3 Model outputting raw hidden-states without any specific head on top.
 */
export class Gemma3PreTrainedModel extends PreTrainedModel {}

/**
 * The bare Gemma3 Model outputting raw hidden-states without any specific head on top.
 */
export class Gemma3Model extends Gemma3PreTrainedModel {}

export class Gemma3ForCausalLM extends Gemma3PreTrainedModel {}
