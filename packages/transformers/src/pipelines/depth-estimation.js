import { Pipeline, prepareImages } from './_base.js';
import { RawImage } from '../utils/image.js';

import { interpolate_4d } from '../utils/tensor.js';

/**
 * @typedef {import('./_base.js').ImagePipelineConstructorArgs} ImagePipelineConstructorArgs
 * @typedef {import('./_base.js').Disposable} Disposable
 * @typedef {import('./_base.js').ImageInput} ImageInput
 */

/**
 * @typedef {Object} DepthEstimationOutput
 * @property {import('../utils/tensor.js').Tensor} predicted_depth The raw depth map predicted by the model.
 * @property {RawImage} depth The processed depth map as an image (with the same size as the input image).
 *
 * @callback DepthEstimationPipelineCallbackSingle Predicts the depth for a single image input.
 * @param {ImageInput} images The image to compute depth for.
 * @returns {Promise<DepthEstimationOutput>} An object containing the depth estimation result.
 *
 * @callback DepthEstimationPipelineCallbackBatch Predicts the depth for multiple image inputs.
 * @param {ImageInput[]} images The images to compute depth for.
 * @returns {Promise<DepthEstimationOutput[]>} A list of objects containing depth estimation results.
 *
 * @typedef {DepthEstimationPipelineCallbackSingle & DepthEstimationPipelineCallbackBatch} DepthEstimationPipelineCallback
 *
 * @typedef {ImagePipelineConstructorArgs & DepthEstimationPipelineCallback & Disposable} DepthEstimationPipelineType
 */

/**
 * Depth estimation pipeline using any `AutoModelForDepthEstimation`. This pipeline predicts the depth of an image.
 *
 * **Example:** Depth estimation w/ `onnx-community/depth-anything-v2-small`
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const depth_estimator = await pipeline('depth-estimation', 'onnx-community/depth-anything-v2-small');
 * const image = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg';
 * const output = await depth_estimator(image);
 * // {
 * //   predicted_depth: Tensor {
 * //     dims: [ 480, 640 ],
 * //     type: 'float32',
 * //     data: Float32Array(307200) [ 2.6300313472747803, 2.5856235027313232, 2.620532751083374, ... ],
 * //     size: 307200
 * //   },
 * //   depth: RawImage {
 * //     data: Uint8Array(307200) [ 106, 104, 106, ... ],
 * //     width: 640,
 * //     height: 480,
 * //     channels: 1
 * //   }
 * // }
 * ```
 */
export class DepthEstimationPipeline
    extends /** @type {new (options: ImagePipelineConstructorArgs) => DepthEstimationPipelineType} */ (Pipeline)
{
    async _call(images) {
        const preparedImages = await prepareImages(images);

        const inputs = await this.processor(preparedImages);
        const { predicted_depth } = await this.model(inputs);

        const toReturn = [];
        for (let i = 0; i < preparedImages.length; ++i) {
            const batch = predicted_depth[i];
            const [height, width] = batch.dims.slice(-2);
            const [new_width, new_height] = preparedImages[i].size;

            // Interpolate to original size
            const prediction = (
                await interpolate_4d(batch.view(1, 1, height, width), {
                    size: [new_height, new_width],
                    mode: 'bilinear',
                })
            ).view(new_height, new_width);

            const minval = /** @type {number} */ (prediction.min().item());
            const maxval = /** @type {number} */ (prediction.max().item());
            const formatted = prediction
                .sub(minval)
                .div_(maxval - minval)
                .mul_(255)
                .to('uint8')
                .unsqueeze(0);
            const depth = RawImage.fromTensor(formatted);
            toReturn.push({
                predicted_depth: prediction,
                depth,
            });
        }
        return Array.isArray(images) ? toReturn : toReturn[0];
    }
}
