import { PreTrainedModel } from '../models.js';
import { SequenceClassifierOutput } from './_base.js';

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
