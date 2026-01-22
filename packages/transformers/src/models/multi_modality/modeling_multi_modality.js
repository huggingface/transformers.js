import { PreTrainedModel, decoder_forward } from '../modeling_utils.js';
import { sessionRun } from '../session.js';
import { pick } from '../../utils/core.js';
import { RawImage } from '../../utils/image.js';
import { Tensor, cat, full, full_like } from '../../utils/tensor.js';

export class MultiModalityPreTrainedModel extends PreTrainedModel {}
export class MultiModalityCausalLM extends MultiModalityPreTrainedModel {
    forward_params = [
        // prepare_inputs_embeds
        'input_ids',
        'pixel_values',
        'images_seq_mask',
        'images_emb_mask',

        // language_model
        'attention_mask',
        'position_ids',
        'past_key_values',
    ];

    /**
     * @param {ConstructorParameters<typeof MultiModalityPreTrainedModel>} args
     */
    constructor(...args) {
        super(...args);

        // State-based approach to switch out which heads to use during generation
        this._generation_mode = 'text';
    }

    async forward(model_inputs) {
        const mode = this._generation_mode ?? 'text';

        // TODO support re-using PKVs for input_ids.dims[1] !== 1
        // if (model_inputs.past_key_values) {
        //     //  && model_inputs.input_ids.dims[1] === 1
        // }

        let output_1;
        if (mode === 'text' || !model_inputs.past_key_values) {
            const session = this.sessions['prepare_inputs_embeds'];
            const prep_inputs = pick(model_inputs, session.inputNames);
            output_1 = await sessionRun(session, prep_inputs);
        } else {
            const session = this.sessions['gen_img_embeds'];
            const prep_inputs = pick(
                {
                    image_ids: model_inputs.input_ids,
                },
                session.inputNames,
            );
            output_1 = await sessionRun(session, prep_inputs);
        }

        const input_2 = { ...model_inputs, ...output_1 };
        const output_2 = await decoder_forward(this, input_2);

        const head = this.sessions[mode === 'text' ? 'lm_head' : 'gen_head'];
        if (!head) {
            throw new Error(`Unable to find "${head}" generation head`);
        }

        const output_3 = await sessionRun(head, pick(output_2, head.inputNames));

        return {
            ...output_1,
            ...output_2,
            ...output_3,
        };
    }

    prepare_inputs_for_generation(input_ids, model_inputs, generation_config) {
        const has_past_key_values = !!model_inputs.past_key_values;

        if (generation_config.guidance_scale !== null && generation_config.guidance_scale > 1) {
            if (has_past_key_values) {
                model_inputs.input_ids = cat([model_inputs.input_ids, model_inputs.input_ids], 0);
                // NOTE: attention_mask handled in generation
            } else {
                model_inputs.input_ids = cat(
                    [model_inputs.input_ids, full_like(model_inputs.input_ids, BigInt(generation_config.pad_token_id))],
                    0,
                );
                model_inputs.attention_mask = cat(
                    [model_inputs.attention_mask, full_like(model_inputs.attention_mask, 0n)],
                    0,
                );
            }
        }

        if (has_past_key_values || !model_inputs.pixel_values) {
            model_inputs.pixel_values = full([0, 0, 3, 384, 384], 1.0);
        }

        if (has_past_key_values) {
            const num_img_tokens = 0;
            const num_text_tokens = 1;
            const has_image = num_img_tokens > 0 ? 1 : 0;

            const batch_size = 1;
            model_inputs.images_seq_mask = new Tensor(
                'bool',
                new Array(num_img_tokens + num_text_tokens).fill(true).fill(false, 0, num_text_tokens),
                [batch_size, num_img_tokens + num_text_tokens],
            );
            model_inputs.images_emb_mask = new Tensor('bool', new Array(num_img_tokens).fill(!!has_image), [
                batch_size,
                1,
                num_img_tokens,
            ]);
        }
        return model_inputs;
    }

    /**
     * @param {import('../../generation/parameters.js').GenerationFunctionParameters} options
     */
    async generate(options) {
        this._generation_mode = 'text';
        return super.generate(options);
    }

    /**
     * @param {import('../../generation/parameters.js').GenerationFunctionParameters} options
     */
    async generate_images(options) {
        this._generation_mode = 'image';

        const start_num_tokens = (options.inputs ?? options[this.main_input_name]).dims[1];
        const all_tokens = await super.generate(options);

        const generated_tokens = /** @type {Tensor} */ (all_tokens).slice(null, [start_num_tokens, null]);

        const image_decode = this.sessions['image_decode'];
        const { decoded_image } = await sessionRun(image_decode, {
            generated_tokens,
        });

        // Equivalent to `np.clip((dec + 1) / 2 * 255, 0, 255)`
        const clamped = decoded_image
            .add_(1)
            .mul_(255 / 2)
            .clamp_(0, 255)
            .to('uint8');

        // Return as a list of images
        const images = [];
        for (const tensor of clamped) {
            const img = RawImage.fromTensor(tensor);
            images.push(img);
        }
        return images;
    }
}
