import { PreTrainedModel } from '../pre-trained-model.js';
import { RTDetrObjectDetectionOutput } from './RTDetrPreTrainedModel.js';

export class DFinePreTrainedModel extends PreTrainedModel {}
export class DFineModel extends DFinePreTrainedModel {}
export class DFineForObjectDetection extends DFinePreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new RTDetrObjectDetectionOutput(await super._call(model_inputs));
    }
}
