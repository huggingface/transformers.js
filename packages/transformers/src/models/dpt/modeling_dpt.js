import { PreTrainedModel } from '../modeling_utils.js';

export class DPTPreTrainedModel extends PreTrainedModel {}

/**
 * The bare DPT Model transformer outputting raw hidden-states without any specific head on top.
 */
export class DPTModel extends DPTPreTrainedModel {}

/**
 * DPT Model with a depth estimation head on top (consisting of 3 convolutional layers) e.g. for KITTI, NYUv2.
 *
 * **Example:** Depth estimation w/ `Xenova/dpt-hybrid-midas`.
 * ```javascript
 * import { DPTForDepthEstimation, AutoProcessor, RawImage, interpolate_4d } from '@huggingface/transformers';
 *
 * // Load model and processor
 * const model_id = 'Xenova/dpt-hybrid-midas';
 * const model = await DPTForDepthEstimation.from_pretrained(model_id);
 * const processor = await AutoProcessor.from_pretrained(model_id);
 *
 * // Load image from URL
 * const url = 'http://images.cocodataset.org/val2017/000000039769.jpg';
 * const image = await RawImage.read(url);
 *
 * // Prepare image for the model
 * const inputs = await processor(image);
 *
 * // Run model
 * const { predicted_depth } = await model(inputs);
 *
 * // Interpolate to original size
 * const prediction = (await interpolate_4d(predicted_depth.unsqueeze(1), {
 *   size: image.size.reverse(),
 *   mode: 'bilinear',
 * })).squeeze(1);
 *
 * // Visualize the prediction
 * const min = prediction.min().item();
 * const max = prediction.max().item();
 * const formatted = prediction.sub_(min).div_(max - min).mul_(255).to('uint8');
 * const depth = RawImage.fromTensor(formatted);
 * // RawImage {
 * //   data: Uint8Array(307200) [ 85, 85, 84, ... ],
 * //   width: 640,
 * //   height: 480,
 * //   channels: 1
 * // }
 * ```
 */
export class DPTForDepthEstimation extends DPTPreTrainedModel {}
