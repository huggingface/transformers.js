import { PreTrainedModel } from '../modeling_utils.js';
import { SequenceClassifierOutput } from '../modeling_outputs.js';

export class IJepaPreTrainedModel extends PreTrainedModel {}
export class IJepaModel extends IJepaPreTrainedModel {}
export class IJepaForImageClassification extends IJepaPreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
