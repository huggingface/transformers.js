import { PreTrainedModel } from '../models.js';

export class TrOCRPreTrainedModel extends PreTrainedModel {}

/**
 * The TrOCR Decoder with a language modeling head.
 */
export class TrOCRForCausalLM extends TrOCRPreTrainedModel {}
