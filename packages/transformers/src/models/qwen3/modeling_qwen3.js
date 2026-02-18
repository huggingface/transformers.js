import { PreTrainedModel } from '../modeling_utils.js';

/**
 * The bare Qwen3 Model outputting raw hidden-states without any specific head on top.
 */
export class Qwen3PreTrainedModel extends PreTrainedModel {}

/**
 * The bare Qwen3 Model outputting raw hidden-states without any specific head on top.
 */
export class Qwen3Model extends Qwen3PreTrainedModel {}

export class Qwen3ForCausalLM extends Qwen3PreTrainedModel {}
