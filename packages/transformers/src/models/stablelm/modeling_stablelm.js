import { PreTrainedModel } from '../modeling_utils.js';

export class StableLmPreTrainedModel extends PreTrainedModel {}

/**
 * The bare StableLm Model transformer outputting raw hidden-states without any specific head on top.
 */
export class StableLmModel extends StableLmPreTrainedModel {}

/**
 * StableLm Model with a `language modeling` head on top for Causal Language Modeling (with past).
 */
export class StableLmForCausalLM extends StableLmPreTrainedModel {}
