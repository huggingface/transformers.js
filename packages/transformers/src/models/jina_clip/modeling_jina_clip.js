import { PreTrainedModel } from '../modeling_utils.js';
import { ones, full } from '../../utils/tensor.js';

export class JinaCLIPPreTrainedModel extends PreTrainedModel {}

export class JinaCLIPModel extends JinaCLIPPreTrainedModel {
    async forward(model_inputs) {
        const missing_text_inputs = !model_inputs.input_ids;
        const missing_image_inputs = !model_inputs.pixel_values;

        if (missing_text_inputs && missing_image_inputs) {
            throw new Error('Either `input_ids` or `pixel_values` should be provided.');
        }

        // If either `input_ids` or `pixel_values` aren't passed, we need to create dummy input since the model requires a value to be specified.
        if (missing_text_inputs) {
            // NOTE: We cannot pass zero-dimension tensor as input for input_ids.
            // Fortunately, the majority of time is spent in the vision encoder, so this shouldn't significantly impact performance.
            model_inputs.input_ids = ones([model_inputs.pixel_values.dims[0], 1]);
        }

        if (missing_image_inputs) {
            // NOTE: Since we create a zero-sized tensor, this does not increase computation time.
            // @ts-ignore
            const { image_size } = this.config.vision_config;
            model_inputs.pixel_values = full([0, 3, image_size, image_size], 0.0); // (pass zero-dimension tensor)
        }

        const { text_embeddings, image_embeddings, l2norm_text_embeddings, l2norm_image_embeddings } =
            await super.forward(model_inputs);

        const result = {};
        if (!missing_text_inputs) {
            result.text_embeddings = text_embeddings;
            result.l2norm_text_embeddings = l2norm_text_embeddings;
        }
        if (!missing_image_inputs) {
            result.image_embeddings = image_embeddings;
            result.l2norm_image_embeddings = l2norm_image_embeddings;
        }
        return result;
    }
}

export class JinaCLIPTextModel extends JinaCLIPPreTrainedModel {
    /** @type {typeof PreTrainedModel.from_pretrained} */
    static async from_pretrained(pretrained_model_name_or_path, options = {}) {
        return super.from_pretrained(pretrained_model_name_or_path, {
            ...options,
            // Update default model file name if not provided
            model_file_name: options.model_file_name ?? 'text_model',
        });
    }
}

export class JinaCLIPVisionModel extends JinaCLIPPreTrainedModel {
    /** @type {typeof PreTrainedModel.from_pretrained} */
    static async from_pretrained(pretrained_model_name_or_path, options = {}) {
        return super.from_pretrained(pretrained_model_name_or_path, {
            ...options,
            // Update default model file name if not provided
            model_file_name: options.model_file_name ?? 'vision_model',
        });
    }
}
