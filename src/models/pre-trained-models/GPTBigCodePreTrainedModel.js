import { PreTrainedModel } from '../pre-trained-model.js';

export class GPTBigCodePreTrainedModel extends PreTrainedModel {}
export class GPTBigCodeModel extends GPTBigCodePreTrainedModel {}

export class GPTBigCodeForCausalLM extends GPTBigCodePreTrainedModel {}
