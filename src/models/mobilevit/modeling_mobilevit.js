import { PreTrainedModel } from '../modeling_utils.js';
import { SequenceClassifierOutput } from '../modeling_outputs.js';

export class MobileViTPreTrainedModel extends PreTrainedModel {}
export class MobileViTModel extends MobileViTPreTrainedModel {}
export class MobileViTForImageClassification extends MobileViTPreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
// TODO: MobileViTForSemanticSegmentation
