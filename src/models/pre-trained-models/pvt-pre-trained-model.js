import { PreTrainedModel } from '../pre-trained-model.js';
import { SequenceClassifierOutput } from '../output.js';

export class PvtPreTrainedModel extends PreTrainedModel {}
export class PvtModel extends PvtPreTrainedModel {}
export class PvtForImageClassification extends PvtPreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
