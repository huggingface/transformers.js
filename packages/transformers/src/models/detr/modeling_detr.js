import { PreTrainedModel } from '../modeling_utils.js';
import { ModelOutput } from '../modeling_outputs.js';
import { Tensor } from '../../utils/tensor.js';

export class DetrPreTrainedModel extends PreTrainedModel {}
export class DetrModel extends DetrPreTrainedModel {}
export class DetrForObjectDetection extends DetrPreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new DetrObjectDetectionOutput(await super._call(model_inputs));
    }
}

export class DetrForSegmentation extends DetrPreTrainedModel {
    /**
     * Runs the model with the provided inputs
     * @param {Object} model_inputs Model inputs
     * @returns {Promise<DetrSegmentationOutput>} Object containing segmentation outputs
     */
    async _call(model_inputs) {
        return new DetrSegmentationOutput(await super._call(model_inputs));
    }
}

export class DetrObjectDetectionOutput extends ModelOutput {
    /**
     * @param {Object} output The output of the model.
     * @param {Tensor} output.logits Classification logits (including no-object) for all queries.
     * @param {Tensor} output.pred_boxes Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height).
     * These values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding possible padding).
     */
    constructor({ logits, pred_boxes }) {
        super();
        this.logits = logits;
        this.pred_boxes = pred_boxes;
    }
}

export class DetrSegmentationOutput extends ModelOutput {
    /**
     * @param {Object} output The output of the model.
     * @param {Tensor} output.logits The output logits of the model.
     * @param {Tensor} output.pred_boxes Predicted boxes.
     * @param {Tensor} output.pred_masks Predicted masks.
     */
    constructor({ logits, pred_boxes, pred_masks }) {
        super();
        this.logits = logits;
        this.pred_boxes = pred_boxes;
        this.pred_masks = pred_masks;
    }
}
