import { PreTrainedModel } from '../pre-trained-model.js';
import { ones, full, randn } from '../../utils/tensor.js';
import { sessionRun } from '../session.js';

export class SupertonicPreTrainedModel extends PreTrainedModel {}
export class SupertonicForConditionalGeneration extends SupertonicPreTrainedModel {
    async generate_speech({
        // Required inputs
        input_ids,
        attention_mask,
        style,

        // Optional inputs
        num_inference_steps = 5,
        speed = 1.05,
    }) {
        // @ts-expect-error TS2339
        const { sampling_rate, chunk_compress_factor, base_chunk_size, latent_dim } = this.config;

        // 1. Text Encoder
        const { last_hidden_state, durations } = await sessionRun(this.sessions['text_encoder'], {
            input_ids,
            attention_mask,
            style,
        });
        durations.div_(speed); // Apply speed factor to duration

        // 2. Latent Denoiser
        const wav_len_max = durations.max().item() * sampling_rate;
        const chunk_size = base_chunk_size * chunk_compress_factor;
        const latent_len = Math.floor((wav_len_max + chunk_size - 1) / chunk_size);
        const batch_size = input_ids.dims[0];
        const latent_mask = ones([batch_size, latent_len]);
        const num_steps = full([batch_size], num_inference_steps);

        let noisy_latents = randn([batch_size, latent_dim * chunk_compress_factor, latent_len]);
        for (let step = 0; step < num_inference_steps; ++step) {
            const timestep = full([batch_size], step);
            ({ denoised_latents: noisy_latents } = await sessionRun(this.sessions['latent_denoiser'], {
                style,
                noisy_latents,
                latent_mask,
                encoder_outputs: last_hidden_state,
                attention_mask,
                timestep,
                num_inference_steps: num_steps,
            }));
        }

        // 3. Voice Decoder
        const { waveform } = await sessionRun(this.sessions['voice_decoder'], {
            latents: noisy_latents,
        });
        return {
            waveform,
            durations,
        };
    }
}
