import { PreTrainedModel } from '../modeling_utils.js';
import { RTDetrObjectDetectionOutput } from '../rt_detr/modeling_rt_detr.js';

export class RFDetrPreTrainedModel extends PreTrainedModel {}
export class RFDetrModel extends RFDetrPreTrainedModel {}
export class RFDetrForObjectDetection extends RFDetrPreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new RFDetrObjectDetectionOutput(await super._call(model_inputs));
    }
}

export class RFDetrObjectDetectionOutput extends RTDetrObjectDetectionOutput {}
