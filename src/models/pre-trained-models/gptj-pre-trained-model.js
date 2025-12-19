import { PreTrainedModel } from '../pre-trained-model.js';

export class GPTJPreTrainedModel extends PreTrainedModel {}
export class GPTJModel extends GPTJPreTrainedModel {}

export class GPTJForCausalLM extends GPTJPreTrainedModel {}
