import { PreTrainedModel } from '../modeling_utils.js';

export class CLIPPreTrainedModel extends PreTrainedModel {}

/**
 * CLIP Text and Vision Model with a projection layers on top
 *
 * **Example:** Perform zero-shot image classification with a `CLIPModel`.
 *
 * ```javascript
 * import { AutoTokenizer, AutoProcessor, CLIPModel, RawImage } from '@huggingface/transformers';
 *
 * // Load tokenizer, processor, and model
 * const tokenizer = await AutoTokenizer.from_pretrained('Xenova/clip-vit-base-patch16');
 * const processor = await AutoProcessor.from_pretrained('Xenova/clip-vit-base-patch16');
 * const model = await CLIPModel.from_pretrained('Xenova/clip-vit-base-patch16');
 *
 * // Run tokenization
 * const texts = ['a photo of a car', 'a photo of a football match']
 * const text_inputs = tokenizer(texts, { padding: true, truncation: true });
 *
 * // Read image and run processor
 * const image = await RawImage.read('https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/football-match.jpg');
 * const image_inputs = await processor(image);
 *
 * // Run model with both text and pixel inputs
 * const output = await model({ ...text_inputs, ...image_inputs });
 * // {
 * //   logits_per_image: Tensor {
 * //     dims: [ 1, 2 ],
 * //     data: Float32Array(2) [ 18.579734802246094, 24.31830596923828 ],
 * //   },
 * //   logits_per_text: Tensor {
 * //     dims: [ 2, 1 ],
 * //     data: Float32Array(2) [ 18.579734802246094, 24.31830596923828 ],
 * //   },
 * //   text_embeds: Tensor {
 * //     dims: [ 2, 512 ],
 * //     data: Float32Array(1024) [ ... ],
 * //   },
 * //   image_embeds: Tensor {
 * //     dims: [ 1, 512 ],
 * //     data: Float32Array(512) [ ... ],
 * //   }
 * // }
 * ```
 */
export class CLIPModel extends CLIPPreTrainedModel {}

/**
 * The text model from CLIP without any head or projection on top.
 */
export class CLIPTextModel extends CLIPPreTrainedModel {
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
 * CLIP Text Model with a projection layer on top (a linear layer on top of the pooled output)
 *
 * **Example:** Compute text embeddings with `CLIPTextModelWithProjection`.
 *
 * ```javascript
 * import { AutoTokenizer, CLIPTextModelWithProjection } from '@huggingface/transformers';
 *
 * // Load tokenizer and text model
 * const tokenizer = await AutoTokenizer.from_pretrained('Xenova/clip-vit-base-patch16');
 * const text_model = await CLIPTextModelWithProjection.from_pretrained('Xenova/clip-vit-base-patch16');
 *
 * // Run tokenization
 * const texts = ['a photo of a car', 'a photo of a football match'];
 * const text_inputs = tokenizer(texts, { padding: true, truncation: true });
 *
 * // Compute embeddings
 * const { text_embeds } = await text_model(text_inputs);
 * // Tensor {
 * //   dims: [ 2, 512 ],
 * //   type: 'float32',
 * //   data: Float32Array(1024) [ ... ],
 * //   size: 1024
 * // }
 * ```
 */
export class CLIPTextModelWithProjection extends CLIPPreTrainedModel {
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
 * The vision model from CLIP without any head or projection on top.
 */
export class CLIPVisionModel extends CLIPPreTrainedModel {
    /** @type {typeof PreTrainedModel.from_pretrained} */
    static async from_pretrained(pretrained_model_name_or_path, options = {}) {
        return super.from_pretrained(pretrained_model_name_or_path, {
            ...options,
            // Update default model file name if not provided
            model_file_name: options.model_file_name ?? 'vision_model',
        });
    }
}

/**
 * CLIP Vision Model with a projection layer on top (a linear layer on top of the pooled output)
 *
 * **Example:** Compute vision embeddings with `CLIPVisionModelWithProjection`.
 *
 * ```javascript
 * import { AutoProcessor, CLIPVisionModelWithProjection, RawImage } from '@huggingface/transformers';
 *
 * // Load processor and vision model
 * const processor = await AutoProcessor.from_pretrained('Xenova/clip-vit-base-patch16');
 * const vision_model = await CLIPVisionModelWithProjection.from_pretrained('Xenova/clip-vit-base-patch16');
 *
 * // Read image and run processor
 * const image = await RawImage.read('https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/football-match.jpg');
 * const image_inputs = await processor(image);
 *
 * // Compute embeddings
 * const { image_embeds } = await vision_model(image_inputs);
 * // Tensor {
 * //   dims: [ 1, 512 ],
 * //   type: 'float32',
 * //   data: Float32Array(512) [ ... ],
 * //   size: 512
 * // }
 * ```
 */
export class CLIPVisionModelWithProjection extends CLIPPreTrainedModel {
    /** @type {typeof PreTrainedModel.from_pretrained} */
    static async from_pretrained(pretrained_model_name_or_path, options = {}) {
        return super.from_pretrained(pretrained_model_name_or_path, {
            ...options,
            // Update default model file name if not provided
            model_file_name: options.model_file_name ?? 'vision_model',
        });
    }
}
