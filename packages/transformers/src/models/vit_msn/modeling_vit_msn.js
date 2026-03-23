import { PreTrainedModel } from '../modeling_utils.js';
import { SequenceClassifierOutput } from '../modeling_outputs.js';

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
