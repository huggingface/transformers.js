import { PreTrainedModel } from '../modeling_utils.js';
/**
 * An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained models.
 */
export class LongT5PreTrainedModel extends PreTrainedModel {}

/**
 * The bare LONGT5 Model transformer outputting raw hidden-states without any specific head on top.
 */
export class LongT5Model extends LongT5PreTrainedModel {}

/**
 * LONGT5 Model with a `language modeling` head on top.
 */
export class LongT5ForConditionalGeneration extends LongT5PreTrainedModel {}
