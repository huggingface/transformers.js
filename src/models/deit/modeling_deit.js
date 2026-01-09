import { PreTrainedModel } from '../_base/pre-trained-model.js';
import { SequenceClassifierOutput } from '../_base/output.js';

export class DeiTPreTrainedModel extends PreTrainedModel {}
export class DeiTModel extends DeiTPreTrainedModel {}
export class DeiTForImageClassification extends DeiTPreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
