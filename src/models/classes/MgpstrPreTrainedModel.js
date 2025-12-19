import { PreTrainedModel } from '../models.js';
import { MgpstrModelOutput } from './_base.js';

export class MgpstrPreTrainedModel extends PreTrainedModel {}

/**
 * MGP-STR Model transformer with three classification heads on top
 * (three A^3 modules and three linear layer on top of the transformer encoder output) for scene text recognition (STR).
 */
export class MgpstrForSceneTextRecognition extends MgpstrPreTrainedModel {
    /**
     * @param {any} model_inputs
     */
    async _call(model_inputs) {
        return new MgpstrModelOutput(await super._call(model_inputs));
    }
}
