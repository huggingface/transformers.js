import { PreTrainedModel } from '../modeling_utils.js';

export class JAISPreTrainedModel extends PreTrainedModel {}

/**
 * The bare JAIS Model transformer outputting raw hidden-states without any specific head on top.
 */
export class JAISModel extends JAISPreTrainedModel {}

/**
 * The JAIS Model transformer with a language modeling head on top (linear layer with weights tied to the input embeddings).
 */
export class JAISLMHeadModel extends JAISPreTrainedModel {}
