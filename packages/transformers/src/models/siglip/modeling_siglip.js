import { PreTrainedModel } from '../modeling_utils.js';
import { CLIPPreTrainedModel } from '../clip/modeling_clip.js';

export class SiglipPreTrainedModel extends PreTrainedModel {}

/**
 * SigLIP Text and Vision Model with a projection layers on top
 *
 * **Example:** Perform zero-shot image classification with a `SiglipModel`.
 *
 * ```javascript
 * import { AutoTokenizer, AutoProcessor, SiglipModel, RawImage } from '@huggingface/transformers';
 *
 * // Load tokenizer, processor, and model
 * const tokenizer = await AutoTokenizer.from_pretrained('Xenova/siglip-base-patch16-224');
 * const processor = await AutoProcessor.from_pretrained('Xenova/siglip-base-patch16-224');
 * const model = await SiglipModel.from_pretrained('Xenova/siglip-base-patch16-224');
 *
 * // Run tokenization
 * const texts = ['a photo of 2 cats', 'a photo of 2 dogs'];
 * const text_inputs = tokenizer(texts, { padding: 'max_length', truncation: true });
 *
 * // Read image and run processor
 * const image = await RawImage.read('http://images.cocodataset.org/val2017/000000039769.jpg');
 * const image_inputs = await processor(image);
 *
 * // Run model with both text and pixel inputs
 * const output = await model({ ...text_inputs, ...image_inputs });
 * // {
 * //   logits_per_image: Tensor {
 * //     dims: [ 1, 2 ],
 * //     data: Float32Array(2) [ -1.6019744873046875, -10.720091819763184 ],
 * //   },
 * //   logits_per_text: Tensor {
 * //     dims: [ 2, 1 ],
 * //     data: Float32Array(2) [ -1.6019744873046875, -10.720091819763184 ],
 * //   },
 * //   text_embeds: Tensor {
 * //     dims: [ 2, 768 ],
 * //     data: Float32Array(1536) [ ... ],
 * //   },
 * //   image_embeds: Tensor {
 * //     dims: [ 1, 768 ],
 * //     data: Float32Array(768) [ ... ],
 * //   }
 * // }
 * ```
 */
export class SiglipModel extends SiglipPreTrainedModel {}

/**
 * The text model from SigLIP without any head or projection on top.
 *
 * **Example:** Compute text embeddings with `SiglipTextModel`.
 *
 * ```javascript
 * import { AutoTokenizer, SiglipTextModel } from '@huggingface/transformers';
 *
 * // Load tokenizer and text model
 * const tokenizer = await AutoTokenizer.from_pretrained('Xenova/siglip-base-patch16-224');
 * const text_model = await SiglipTextModel.from_pretrained('Xenova/siglip-base-patch16-224');
 *
 * // Run tokenization
 * const texts = ['a photo of 2 cats', 'a photo of 2 dogs'];
 * const text_inputs = tokenizer(texts, { padding: 'max_length', truncation: true });
 *
 * // Compute embeddings
 * const { pooler_output } = await text_model(text_inputs);
 * // Tensor {
 * //   dims: [ 2, 768 ],
 * //   type: 'float32',
 * //   data: Float32Array(1536) [ ... ],
 * //   size: 1536
 * // }
 * ```
 */
export class SiglipTextModel extends SiglipPreTrainedModel {
    /** @type {typeof PreTrainedModel.from_pretrained} */
    static async from_pretrained(pretrained_model_name_or_path, options = {}) {
        return super.from_pretrained(pretrained_model_name_or_path, {
            ...options,
            // Update default model file name if not provided
            model_file_name: options.model_file_name ?? 'text_model',
        });
    }
}

/**
 * The vision model from SigLIP without any head or projection on top.
 *
 * **Example:** Compute vision embeddings with `SiglipVisionModel`.
 *
 * ```javascript
 * import { AutoProcessor, SiglipVisionModel, RawImage } from '@huggingface/transformers';
 *
 * // Load processor and vision model
 * const processor = await AutoProcessor.from_pretrained('Xenova/siglip-base-patch16-224');
 * const vision_model = await SiglipVisionModel.from_pretrained('Xenova/siglip-base-patch16-224');
 *
 * // Read image and run processor
 * const image = await RawImage.read('https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/football-match.jpg');
 * const image_inputs = await processor(image);
 *
 * // Compute embeddings
 * const { pooler_output } = await vision_model(image_inputs);
 * // Tensor {
 * //   dims: [ 1, 768 ],
 * //   type: 'float32',
 * //   data: Float32Array(768) [ ... ],
 * //   size: 768
 * // }
 * ```
 */
export class SiglipVisionModel extends CLIPPreTrainedModel {
    /** @type {typeof PreTrainedModel.from_pretrained} */
    static async from_pretrained(pretrained_model_name_or_path, options = {}) {
        return super.from_pretrained(pretrained_model_name_or_path, {
            ...options,
            // Update default model file name if not provided
            model_file_name: options.model_file_name ?? 'vision_model',
        });
    }
}
