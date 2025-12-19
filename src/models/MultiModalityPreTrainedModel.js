import { PreTrainedModel } from '../models.js';
import { sessionRun } from './session.js';
import { decoderForward } from './utils.js';
import { pick } from '../utils/core.js';
import { RawImage } from '../utils/image.js';

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
        const output_2 = await decoderForward(this, input_2);

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

    /**
     * @param {import('./generation/parameters.js').GenerationFunctionParameters} options
     */
    async generate(options) {
        this._generation_mode = 'text';
        return super.generate(options);
    }

    /**
     * @param {import('./generation/parameters.js').GenerationFunctionParameters} options
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
