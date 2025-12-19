import { PreTrainedModel } from '../models.js';

export class GPTBigCodePreTrainedModel extends PreTrainedModel {}
export class GPTBigCodeModel extends GPTBigCodePreTrainedModel {}

export class GPTBigCodeForCausalLM extends GPTBigCodePreTrainedModel {}
