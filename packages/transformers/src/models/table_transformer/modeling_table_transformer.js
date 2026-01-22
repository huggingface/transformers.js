import { PreTrainedModel } from '../modeling_utils.js';
import { DetrObjectDetectionOutput } from '../detr/modeling_detr.js';

export class TableTransformerPreTrainedModel extends PreTrainedModel {}

/**
 * The bare Table Transformer Model (consisting of a backbone and encoder-decoder Transformer)
 * outputting raw hidden-states without any specific head on top.
 */
export class TableTransformerModel extends TableTransformerPreTrainedModel {}

/**
 * Table Transformer Model (consisting of a backbone and encoder-decoder Transformer)
 * with object detection heads on top, for tasks such as COCO detection.
 */
export class TableTransformerForObjectDetection extends TableTransformerPreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new TableTransformerObjectDetectionOutput(await super._call(model_inputs));
    }
}
export class TableTransformerObjectDetectionOutput extends DetrObjectDetectionOutput {}
