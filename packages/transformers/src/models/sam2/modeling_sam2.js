import { PreTrainedModel, encoder_forward } from '../modeling_utils.js';
import { sessionRun } from '../session.js';
import { ModelOutput } from '../modeling_outputs.js';
import { ones, full } from '../../utils/tensor.js';
import { pick } from '../../utils/core.js';
import { Tensor } from '../../utils/tensor.js';

export class Sam2ImageSegmentationOutput extends ModelOutput {
    /**
     * @param {Object} output The output of the model.
     * @param {Tensor} output.iou_scores The output logits of the model.
     * @param {Tensor} output.pred_masks Predicted boxes.
     * @param {Tensor} output.object_score_logits Logits for the object score, indicating if an object is present.
     */
    constructor({ iou_scores, pred_masks, object_score_logits }) {
        super();
        this.iou_scores = iou_scores;
        this.pred_masks = pred_masks;
        this.object_score_logits = object_score_logits;
    }
}

export class Sam2PreTrainedModel extends PreTrainedModel {}
export class Sam2Model extends Sam2PreTrainedModel {
    /**
     * Compute image embeddings and positional image embeddings, given the pixel values of an image.
     * @param {Object} model_inputs Object containing the model inputs.
     * @param {Tensor} model_inputs.pixel_values Pixel values obtained using a `Sam2Processor`.
     * @returns {Promise<Record<String, Tensor>>} The image embeddings.
     */
    async get_image_embeddings({ pixel_values }) {
        // in:
        //  - pixel_values: tensor.float32[batch_size,3,1024,1024]
        //
        // out:
        //  - image_embeddings.0: tensor.float32[batch_size,32,256,256]
        //  - image_embeddings.1: tensor.float32[batch_size,64,128,128]
        //  - image_embeddings.2: tensor.float32[batch_size,256,64,64]
        return await encoder_forward(this, { pixel_values });
    }

    async forward(model_inputs) {
        // @ts-expect-error ts(2339)
        const { num_feature_levels } = this.config.vision_config;
        const image_embeddings_name = Array.from({ length: num_feature_levels }, (_, i) => `image_embeddings.${i}`);

        if (image_embeddings_name.some((name) => !model_inputs[name])) {
            // Compute the image embeddings if they are missing
            model_inputs = {
                ...model_inputs,
                ...(await this.get_image_embeddings(model_inputs)),
            };
        } else {
            model_inputs = { ...model_inputs };
        }

        if (model_inputs.input_points) {
            if (model_inputs.input_boxes && model_inputs.input_boxes.dims[1] !== 1) {
                throw new Error(
                    'When both `input_points` and `input_boxes` are provided, the number of boxes per image must be 1.',
                );
            }
            const shape = model_inputs.input_points.dims;
            model_inputs.input_labels ??= ones(shape.slice(0, -1));
            model_inputs.input_boxes ??= full([shape[0], 0, 4], 0.0);
        } else if (model_inputs.input_boxes) {
            // only boxes
            const shape = model_inputs.input_boxes.dims;
            model_inputs.input_labels = full([shape[0], shape[1], 0], -1n);
            model_inputs.input_points = full([shape[0], 1, 0, 2], 0.0);
        } else {
            throw new Error('At least one of `input_points` or `input_boxes` must be provided.');
        }

        const prompt_encoder_mask_decoder_session = this.sessions['prompt_encoder_mask_decoder'];
        const decoder_inputs = pick(model_inputs, prompt_encoder_mask_decoder_session.inputNames);

        // Returns:
        //  - iou_scores: tensor.float32[batch_size,num_boxes_or_points,3]
        //  - pred_masks: tensor.float32[batch_size,num_boxes_or_points,3,256,256]
        //  - object_score_logits: tensor.float32[batch_size,num_boxes_or_points,1]
        return await sessionRun(prompt_encoder_mask_decoder_session, decoder_inputs);
    }

    /**
     * Runs the model with the provided inputs
     * @param {Object} model_inputs Model inputs
     * @returns {Promise<Sam2ImageSegmentationOutput>} Object containing segmentation outputs
     */
    async _call(model_inputs) {
        return new Sam2ImageSegmentationOutput(await super._call(model_inputs));
    }
}
export class EdgeTamModel extends Sam2Model {} // NOTE: extends Sam2Model
export class Sam3TrackerModel extends Sam2Model {} // NOTE: extends Sam2Model
