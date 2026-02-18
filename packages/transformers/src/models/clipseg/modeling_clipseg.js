import { PreTrainedModel } from '../modeling_utils.js';

export class CLIPSegPreTrainedModel extends PreTrainedModel {}

export class CLIPSegModel extends CLIPSegPreTrainedModel {}

/**
 * CLIPSeg model with a Transformer-based decoder on top for zero-shot and one-shot image segmentation.
 *
 * **Example:** Perform zero-shot image segmentation with a `CLIPSegForImageSegmentation` model.
 *
 * ```javascript
 * import { AutoTokenizer, AutoProcessor, CLIPSegForImageSegmentation, RawImage } from '@huggingface/transformers';
 *
 * // Load tokenizer, processor, and model
 * const tokenizer = await AutoTokenizer.from_pretrained('Xenova/clipseg-rd64-refined');
 * const processor = await AutoProcessor.from_pretrained('Xenova/clipseg-rd64-refined');
 * const model = await CLIPSegForImageSegmentation.from_pretrained('Xenova/clipseg-rd64-refined');
 *
 * // Run tokenization
 * const texts = ['a glass', 'something to fill', 'wood', 'a jar'];
 * const text_inputs = tokenizer(texts, { padding: true, truncation: true });
 *
 * // Read image and run processor
 * const image = await RawImage.read('https://github.com/timojl/clipseg/blob/master/example_image.jpg?raw=true');
 * const image_inputs = await processor(image);
 *
 * // Run model with both text and pixel inputs
 * const { logits } = await model({ ...text_inputs, ...image_inputs });
 * // logits: Tensor {
 * //   dims: [4, 352, 352],
 * //   type: 'float32',
 * //   data: Float32Array(495616) [ ... ],
 * //   size: 495616
 * // }
 * ```
 *
 * You can visualize the predictions as follows:
 * ```javascript
 * const preds = logits
 *   .unsqueeze_(1)
 *   .sigmoid_()
 *   .mul_(255)
 *   .round_()
 *   .to('uint8');
 *
 * for (let i = 0; i < preds.dims[0]; ++i) {
 *   const img = RawImage.fromTensor(preds[i]);
 *   img.save(`prediction_${i}.png`);
 * }
 * ```
 */
export class CLIPSegForImageSegmentation extends CLIPSegPreTrainedModel {}
