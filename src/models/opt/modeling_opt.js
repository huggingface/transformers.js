import { PreTrainedModel } from '../modeling_utils.js';

export class OPTPreTrainedModel extends PreTrainedModel {}

/**
 * The bare OPT Model outputting raw hidden-states without any specific head on top.
 */
export class OPTModel extends OPTPreTrainedModel {}

/**
 * The OPT Model transformer with a language modeling head on top (linear layer with weights tied to the input embeddings).
 */
export class OPTForCausalLM extends OPTPreTrainedModel {}
