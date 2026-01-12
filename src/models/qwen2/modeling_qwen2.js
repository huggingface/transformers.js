import { PreTrainedModel } from '../modeling_utils.js';

/**
 * The bare Qwen2 Model outputting raw hidden-states without any specific head on top.
 */
export class Qwen2PreTrainedModel extends PreTrainedModel {}

/**
 * The bare Qwen2 Model outputting raw hidden-states without any specific head on top.
 */
export class Qwen2Model extends Qwen2PreTrainedModel {}

export class Qwen2ForCausalLM extends Qwen2PreTrainedModel {}
