import { PreTrainedModel } from '../_base/pre-trained-model.js';
import { SequenceClassifierOutput } from '../_base/output.js';

export class HieraPreTrainedModel extends PreTrainedModel {}
export class HieraModel extends HieraPreTrainedModel {}
export class HieraForImageClassification extends HieraPreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
