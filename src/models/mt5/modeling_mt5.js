import { PreTrainedModel } from '../_base/pre-trained-model.js';

export class MT5PreTrainedModel extends PreTrainedModel {}

export class MT5Model extends MT5PreTrainedModel {}

/**
 * A class representing a conditional sequence-to-sequence model based on the MT5 architecture.
 */
export class MT5ForConditionalGeneration extends MT5PreTrainedModel {}