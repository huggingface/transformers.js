import { PreTrainedModel } from '../_base/pre-trained-model.js';
import { SequenceClassifierOutput } from '../_base/output.js';

export class FastViTPreTrainedModel extends PreTrainedModel {}
export class FastViTModel extends FastViTPreTrainedModel {}
export class FastViTForImageClassification extends FastViTPreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
