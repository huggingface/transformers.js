import { PreTrainedModel } from '../modeling_utils.js';

/**
 * The bare LLama Model outputting raw hidden-states without any specific head on top.
 */
export class LlamaPreTrainedModel extends PreTrainedModel {}

/**
 * The bare LLaMA Model outputting raw hidden-states without any specific head on top.
 */
export class LlamaModel extends LlamaPreTrainedModel {}

export class LlamaForCausalLM extends LlamaPreTrainedModel {}
