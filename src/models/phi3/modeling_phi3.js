import { PreTrainedModel } from '../modeling_utils.js';

export class Phi3PreTrainedModel extends PreTrainedModel {}

/**
 * The bare Phi3 Model outputting raw hidden-states without any specific head on top.
 */
export class Phi3Model extends Phi3PreTrainedModel {}

export class Phi3ForCausalLM extends Phi3PreTrainedModel {}
