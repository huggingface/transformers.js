import { PreTrainedModel, encoder_forward } from '../modeling_utils.js';
import { ones } from '../../utils/tensor.js';
import { sessionRun } from '../session.js';
import { ModelOutput } from '../modeling_outputs.js';
import { Tensor } from '../../utils/tensor.js';

/**
 * Base class for Segment-Anything model's output.
 */
export class SamImageSegmentationOutput extends ModelOutput {
    /**
     * @param {Object} output The output of the model.
     * @param {Tensor} output.iou_scores The output logits of the model.
     * @param {Tensor} output.pred_masks Predicted boxes.
     */
    constructor({ iou_scores, pred_masks }) {
        super();
        this.iou_scores = iou_scores;
        this.pred_masks = pred_masks;
    }
}

export class SamPreTrainedModel extends PreTrainedModel {}

/**
 * Segment Anything Model (SAM) for generating segmentation masks, given an input image
 * and optional 2D location and bounding boxes.
 *
 * **Example:** Perform mask generation w/ `Xenova/sam-vit-base`.
 * ```javascript
 * import { SamModel, AutoProcessor, RawImage } from '@huggingface/transformers';
 *
 * const model = await SamModel.from_pretrained('Xenova/sam-vit-base');
 * const processor = await AutoProcessor.from_pretrained('Xenova/sam-vit-base');
 *
 * const img_url = 'https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png';
 * const raw_image = await RawImage.read(img_url);
 * const input_points = [[[450, 600]]] // 2D localization of a window
 *
 * const inputs = await processor(raw_image, { input_points });
 * const outputs = await model(inputs);
 *
 * const masks = await processor.post_process_masks(outputs.pred_masks, inputs.original_sizes, inputs.reshaped_input_sizes);
 * // [
 * //   Tensor {
 * //     dims: [ 1, 3, 1764, 2646 ],
 * //     type: 'bool',
 * //     data: Uint8Array(14002632) [ ... ],
 * //     size: 14002632
 * //   }
 * // ]
 * const scores = outputs.iou_scores;
 * // Tensor {
 * //   dims: [ 1, 1, 3 ],
 * //   type: 'float32',
 * //   data: Float32Array(3) [
 * //     0.8892380595207214,
 * //     0.9311248064041138,
 * //     0.983696699142456
 * //   ],
 * //   size: 3
 * // }
 * ```
 */
export class SamModel extends SamPreTrainedModel {
    /**
     * Compute image embeddings and positional image embeddings, given the pixel values of an image.
     * @param {Object} model_inputs Object containing the model inputs.
     * @param {Tensor} model_inputs.pixel_values Pixel values obtained using a `SamProcessor`.
     * @returns {Promise<{ image_embeddings: Tensor, image_positional_embeddings: Tensor }>} The image embeddings and positional image embeddings.
     */
    async get_image_embeddings({ pixel_values }) {
        // in:
        //  - pixel_values: tensor.float32[batch_size,3,1024,1024]
        //
        // out:
        //  - image_embeddings: tensor.float32[batch_size,256,64,64]
        //  - image_positional_embeddings: tensor.float32[batch_size,256,64,64]
        return await encoder_forward(this, { pixel_values });
    }

    /**
     * @typedef {Object} SamModelInputs Object containing the model inputs.
     * @property {Tensor} pixel_values Pixel values as a Tensor with shape `(batch_size, num_channels, height, width)`.
     * These can be obtained using a `SamProcessor`.
     * @property {Tensor} [input_points] Input 2D spatial points with shape `(batch_size, num_points, 2)`.
     * This is used by the prompt encoder to encode the prompt.
     * @property {Tensor} [input_labels] Input labels for the points, as a Tensor of shape `(batch_size, point_batch_size, num_points)`.
     * This is used by the prompt encoder to encode the prompt. There are 4 types of labels:
     *  - `1`: the point is a point that contains the object of interest
     *  - `0`: the point is a point that does not contain the object of interest
     *  - `-1`: the point corresponds to the background
     *  - `-10`: the point is a padding point, thus should be ignored by the prompt encoder
     * @property {Tensor} [input_boxes] Input bounding boxes with shape `(batch_size, num_boxes, 4)`.
     * @property {Tensor} [image_embeddings] Image embeddings used by the mask decoder.
     * @property {Tensor} [image_positional_embeddings] Image positional embeddings used by the mask decoder.
     */

    /**
     * @param {SamModelInputs} model_inputs Object containing the model inputs.
     * @returns {Promise<Object>} The output of the model.
     */
    async forward(model_inputs) {
        if (!model_inputs.image_embeddings || !model_inputs.image_positional_embeddings) {
            // Compute the image embeddings if they are missing
            model_inputs = {
                ...model_inputs,
                ...(await this.get_image_embeddings(model_inputs)),
            };
        } else {
            model_inputs = { ...model_inputs };
        }

        // Set default input labels if they are missing
        model_inputs.input_labels ??= ones(model_inputs.input_points.dims.slice(0, -1));

        const decoder_inputs = {
            image_embeddings: model_inputs.image_embeddings,
            image_positional_embeddings: model_inputs.image_positional_embeddings,
        };
        if (model_inputs.input_points) {
            decoder_inputs.input_points = model_inputs.input_points;
        }
        if (model_inputs.input_labels) {
            decoder_inputs.input_labels = model_inputs.input_labels;
        }
        if (model_inputs.input_boxes) {
            decoder_inputs.input_boxes = model_inputs.input_boxes;
        }

        // Returns:
        //  - iou_scores: tensor.float32[batch_size,point_batch_size,3]
        //  - pred_masks: tensor.float32[batch_size,point_batch_size,3,256,256]
        return await sessionRun(this.sessions['prompt_encoder_mask_decoder'], decoder_inputs);
    }

    /**
     * Runs the model with the provided inputs
     * @param {Object} model_inputs Model inputs
     * @returns {Promise<SamImageSegmentationOutput>} Object containing segmentation outputs
     */
    async _call(model_inputs) {
        return new SamImageSegmentationOutput(await super._call(model_inputs));
    }
}
