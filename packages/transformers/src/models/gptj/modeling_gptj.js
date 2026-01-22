import { PreTrainedModel } from '../modeling_utils.js';

export class GPTJPreTrainedModel extends PreTrainedModel {}
export class GPTJModel extends GPTJPreTrainedModel {}

export class GPTJForCausalLM extends GPTJPreTrainedModel {}
