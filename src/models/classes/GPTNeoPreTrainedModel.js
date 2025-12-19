import { PreTrainedModel } from '../models.js';

export class GPTNeoPreTrainedModel extends PreTrainedModel {}
export class GPTNeoModel extends GPTNeoPreTrainedModel {}

export class GPTNeoForCausalLM extends GPTNeoPreTrainedModel {}