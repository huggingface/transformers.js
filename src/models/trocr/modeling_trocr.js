import { PreTrainedModel } from '../_base/pre-trained-model.js';

export class TrOCRPreTrainedModel extends PreTrainedModel {}

/**
 * The TrOCR Decoder with a language modeling head.
 */
export class TrOCRForCausalLM extends TrOCRPreTrainedModel {}
