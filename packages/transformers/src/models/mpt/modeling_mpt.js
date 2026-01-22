import { PreTrainedModel } from '../modeling_utils.js';

export class MptPreTrainedModel extends PreTrainedModel {}

/**
 * The bare Mpt Model transformer outputting raw hidden-states without any specific head on top.
 */
export class MptModel extends MptPreTrainedModel {}

/**
 * The MPT Model transformer with a language modeling head on top (linear layer with weights tied to the input embeddings).
 */
export class MptForCausalLM extends MptPreTrainedModel {}
