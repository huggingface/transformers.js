import { PreTrainedModel } from '../modeling_utils.js';

export class ClapPreTrainedModel extends PreTrainedModel {}

export class ClapModel extends ClapPreTrainedModel {}

/**
 * CLAP Text Model with a projection layer on top (a linear layer on top of the pooled output).
 *
 * **Example:** Compute text embeddings with `ClapTextModelWithProjection`.
 *
 * ```javascript
 * import { AutoTokenizer, ClapTextModelWithProjection } from '@huggingface/transformers';
 *
 * // Load tokenizer and text model
 * const tokenizer = await AutoTokenizer.from_pretrained('Xenova/clap-htsat-unfused');
 * const text_model = await ClapTextModelWithProjection.from_pretrained('Xenova/clap-htsat-unfused');
 *
 * // Run tokenization
 * const texts = ['a sound of a cat', 'a sound of a dog'];
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
export class ClapTextModelWithProjection extends ClapPreTrainedModel {
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
 * CLAP Audio Model with a projection layer on top (a linear layer on top of the pooled output).
 *
 * **Example:** Compute audio embeddings with `ClapAudioModelWithProjection`.
 *
 * ```javascript
 * import { AutoProcessor, ClapAudioModelWithProjection, read_audio } from '@huggingface/transformers';
 *
 * // Load processor and audio model
 * const processor = await AutoProcessor.from_pretrained('Xenova/clap-htsat-unfused');
 * const audio_model = await ClapAudioModelWithProjection.from_pretrained('Xenova/clap-htsat-unfused');
 *
 * // Read audio and run processor
 * const audio = await read_audio('https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cat_meow.wav');
 * const audio_inputs = await processor(audio);
 *
 * // Compute embeddings
 * const { audio_embeds } = await audio_model(audio_inputs);
 * // Tensor {
 * //   dims: [ 1, 512 ],
 * //   type: 'float32',
 * //   data: Float32Array(512) [ ... ],
 * //   size: 512
 * // }
 * ```
 */
export class ClapAudioModelWithProjection extends ClapPreTrainedModel {
    /** @type {typeof PreTrainedModel.from_pretrained} */
    static async from_pretrained(pretrained_model_name_or_path, options = {}) {
        return super.from_pretrained(pretrained_model_name_or_path, {
            ...options,
            // Update default model file name if not provided
            model_file_name: options.model_file_name ?? 'audio_model',
        });
    }
}
