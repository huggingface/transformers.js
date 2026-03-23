import { PreTrainedModel } from '../modeling_utils.js';

export class GLPNPreTrainedModel extends PreTrainedModel {}

/**
 * The bare GLPN encoder (Mix-Transformer) outputting raw hidden-states without any specific head on top.
 */
export class GLPNModel extends GLPNPreTrainedModel {}

/**
 * import { GLPNForDepthEstimation, AutoProcessor, RawImage, interpolate_4d } from '@huggingface/transformers';
 *
 * // Load model and processor
 * const model_id = 'Xenova/glpn-kitti';
 * const model = await GLPNForDepthEstimation.from_pretrained(model_id);
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
export class GLPNForDepthEstimation extends GLPNPreTrainedModel {}
