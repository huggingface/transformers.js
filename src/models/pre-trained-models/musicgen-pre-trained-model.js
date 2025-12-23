import { PreTrainedModel } from '../pre-trained-model.js';
import { sessionRun } from '../session.js';
import { Tensor } from '../../utils/tensor.js';
import { ModelOutput } from '../output.js';

export class MusicgenPreTrainedModel extends PreTrainedModel {}

/**
 * The bare Musicgen decoder model outputting raw hidden-states without any specific head on top.
 */
export class MusicgenModel extends MusicgenPreTrainedModel {}

/**
 * The MusicGen decoder model with a language modelling head on top.
 */
export class MusicgenForCausalLM extends MusicgenPreTrainedModel {}

/**
 * The composite MusicGen model with a text encoder, audio encoder and Musicgen decoder,
 * for music generation tasks with one or both of text and audio prompts.
 *
 * **Example:** Generate music from text with `Xenova/musicgen-small`.
 * ```javascript
 * import { AutoTokenizer, MusicgenForConditionalGeneration } from '@huggingface/transformers';
 *
 * // Load tokenizer and model
 * const tokenizer = await AutoTokenizer.from_pretrained('Xenova/musicgen-small');
 * const model = await MusicgenForConditionalGeneration.from_pretrained(
 *   'Xenova/musicgen-small', { dtype: 'fp32' }
 * );
 *
 * // Prepare text input
 * const prompt = '80s pop track with bassy drums and synth';
 * const inputs = tokenizer(prompt);
 *
 * // Generate audio
 * const audio_values = await model.generate({
 *   ...inputs,
 *   max_new_tokens: 512,
 *   do_sample: true,
 *   guidance_scale: 3,
 * });
 *
 * // (Optional) Write the output to a WAV file
 * import wavefile from 'wavefile';
 * import fs from 'fs';
 *
 * const wav = new wavefile.WaveFile();
 * wav.fromScratch(1, model.config.audio_encoder.sampling_rate, '32f', audio_values.data);
 * fs.writeFileSync('musicgen_out.wav', wav.toBuffer());
 * ```
 */
export class MusicgenForConditionalGeneration extends PreTrainedModel {
    // NOTE: not MusicgenPreTrainedModel
    forward_params = [
        'input_ids',
        'attention_mask',
        'encoder_outputs',
        'decoder_input_ids',
        'decoder_attention_mask',
        'past_key_values',
    ];

    /**
     * Apply the pattern mask to the final ids,
     * then revert the pattern delay mask by filtering the pad token id in a single step.
     * @param {Tensor} outputs The output tensor from the model.
     * @returns {Tensor} The filtered output tensor.
     */
    _apply_and_filter_by_delay_pattern_mask(outputs) {
        const [bs_x_codebooks, seqLength] = outputs.dims;
        // @ts-expect-error TS2339
        const num_codebooks = this.config.decoder.num_codebooks;
        const upperBound = seqLength - num_codebooks;

        let newDataSize = 0;
        for (let i = 0; i < outputs.size; ++i) {
            // @ts-expect-error TS2339
            if (outputs.data[i] === this.config.decoder.pad_token_id) {
                continue;
            }

            const row = i % seqLength;
            const col = Math.floor(i / seqLength) % num_codebooks;

            const diff = row - col;
            if (diff > 0 && diff <= upperBound) {
                outputs.data[newDataSize++] = outputs.data[i];
            }
        }

        const batch_size = Math.floor(bs_x_codebooks / num_codebooks);
        const inferred = newDataSize / (batch_size * num_codebooks);
        // TODO: assert `inferred` is an integer
        return new Tensor(outputs.type, outputs.data.slice(0, newDataSize), [batch_size, num_codebooks, inferred]);
    }

    prepare_inputs_for_generation(input_ids, model_inputs, generation_config) {
        // apply the delay pattern mask
        let clonedInputIds = structuredClone(input_ids);
        for (let i = 0; i < clonedInputIds.length; ++i) {
            for (let j = 0; j < clonedInputIds[i].length; ++j) {
                // @ts-expect-error TS2339
                if (i % this.config.decoder.num_codebooks >= j) {
                    // @ts-expect-error TS2339
                    clonedInputIds[i][j] = BigInt(this.config.decoder.pad_token_id);
                }
            }
        }
        // for classifier free guidance we need to replicate the decoder args across the batch dim
        // (we'll split these before sampling)
        if (generation_config.guidance_scale !== null && generation_config.guidance_scale > 1) {
            // [batch, seqLength] -> [2 * batch, seqLength]
            clonedInputIds = clonedInputIds.concat(clonedInputIds);
        }

        const prepped = super.prepare_inputs_for_generation(clonedInputIds, model_inputs, generation_config);
        return prepped;
    }

    /**
     * Generates sequences of token ids for models with a language modeling head.
     * @param {import('../../generation/parameters.js').GenerationFunctionParameters} options
     * @returns {Promise<ModelOutput|Tensor>} The output of the model, which can contain the generated token ids, attentions, and scores.
     */
    async generate(options) {
        const output_ids = await super.generate(options);

        // apply the pattern mask to the final ids
        // tensor: int64[1,batch_size,4,chunk_length]
        const audio_codes = this._apply_and_filter_by_delay_pattern_mask(/** @type {Tensor} */ (output_ids)).unsqueeze_(
            0,
        ); // append the frame dimension back to the audio codes

        const { audio_values } = await sessionRun(this.sessions['encodec_decode'], { audio_codes });

        return audio_values;
    }
}
