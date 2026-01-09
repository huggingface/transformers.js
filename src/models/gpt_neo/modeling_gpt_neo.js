import { PreTrainedModel } from '../_base/pre-trained-model.js';

export class GPTNeoPreTrainedModel extends PreTrainedModel {}
export class GPTNeoModel extends GPTNeoPreTrainedModel {}

export class GPTNeoForCausalLM extends GPTNeoPreTrainedModel {}