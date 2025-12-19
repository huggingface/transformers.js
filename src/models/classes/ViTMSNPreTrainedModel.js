import { PreTrainedModel } from '../models.js';
import { SequenceClassifierOutput } from './_base.js';

export class ViTMSNPreTrainedModel extends PreTrainedModel {}
export class ViTMSNModel extends ViTMSNPreTrainedModel {}
export class ViTMSNForImageClassification extends ViTMSNPreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
