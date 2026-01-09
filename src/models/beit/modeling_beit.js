import { PreTrainedModel } from '../_base/pre-trained-model.js';
import { SequenceClassifierOutput } from '../_base/output.js';

export class BeitPreTrainedModel extends PreTrainedModel {}
export class BeitModel extends BeitPreTrainedModel {}
export class BeitForImageClassification extends BeitPreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
