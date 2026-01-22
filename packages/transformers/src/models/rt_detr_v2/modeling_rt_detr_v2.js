import { PreTrainedModel } from '../modeling_utils.js';
import { RTDetrObjectDetectionOutput } from '../rt_detr/modeling_rt_detr.js';

export class RTDetrV2PreTrainedModel extends PreTrainedModel {}
export class RTDetrV2Model extends RTDetrV2PreTrainedModel {}
export class RTDetrV2ForObjectDetection extends RTDetrV2PreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new RTDetrV2ObjectDetectionOutput(await super._call(model_inputs));
    }
}

export class RTDetrV2ObjectDetectionOutput extends RTDetrObjectDetectionOutput {}
