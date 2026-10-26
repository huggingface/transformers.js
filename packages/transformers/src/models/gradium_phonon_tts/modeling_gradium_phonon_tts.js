/**
 * @module models
 */

import { PreTrainedModel } from '../modeling_utils.js';
import { Tensor, full, randn } from '../../utils/tensor.js';
import { sessionRun } from '../session.js';
import { logger } from '../../utils/logger.js';

const i64 = (values, dims) => new Tensor('int64', BigInt64Array.from(values, BigInt), dims);
const dispose_state = (state) => {
    for (const tensor of Object.values(state ?? {})) tensor.dispose();
};

function empty_cache({ num_layers, num_heads, head_dim }) {
    return Object.fromEntries(
        Array.from({ length: num_layers }, (_, i) =>
            ['key', 'value'].map((key) => [`past_key_values.${i}.${key}`, full([1, num_heads, 0, head_dim], 0)]),
        ).flat(),
    );
}

// Feed the flow model's present KV outputs back under the next step's past input names.
function next_state(outputs) {
    return Object.fromEntries(
        Object.entries(outputs)
            .filter(([name]) => name.startsWith('present.'))
            .map(([name, value]) => [name.replace('present.', 'past_key_values.'), value]),
    );
}

/**
 * The Phonon flow model and audio decoder. The flow model generates one latent per 80 ms frame,
 * then the decoder turns all latents of a chunk into 24 kHz PCM in a single call. Unlike Python,
 * which returns latents for the processor to decode, this returns PCM directly (without end-of-pause trimming).
 * Hub decoder suffixes identify the variant: `mimi_decoder_q4.onnx` is fp32,
 * while `mimi_decoder_q4f16.onnx` is fp16 (only the flow model is quantized).
 */
export class GradiumPhononTTSForConditionalGeneration extends PreTrainedModel {
    // Python keeps these as config values; the checkpoint stores them under `phonon.conditions`.
    get default_conditions() {
        // @ts-expect-error TS2339
        const { num_speakers, duration_delta } = this.config.phonon.conditions;
        return { num_speakers: Number(num_speakers.default), duration_delta: Number(duration_delta.default) };
    }

    async _run_flow(past, { text = [], voice = null, latent = null, noise = null, num_speakers, duration_delta }) {
        // @ts-expect-error TS2339
        const cfg = this.config.phonon;
        const inputs = {
            text_ids: i64(text, [1, text.length]),
            voice_latents: voice ?? full([1, cfg.voice_dim, 0], 0),
            latent: latent ?? full([1, 0, cfg.latent_dim], 0),
            noise: noise ?? full([1, cfg.latent_dim], 0),
            num_speakers: i64([num_speakers], [1]),
            duration_delta: full([1], duration_delta),
        };
        try {
            return await sessionRun(this.sessions.flow_lm, { ...inputs, ...past });
        } finally {
            // The voice and latent belong to the caller; everything else (including noise) is consumed here.
            for (const value of Object.values(inputs)) {
                if (value !== voice && value !== latent) value.dispose();
            }
        }
    }

    // Conditions the flow model on a voice. The returned prefix is owned and freed by `generate_speech`.
    async _prepare_voice(voice) {
        // @ts-expect-error TS2339
        const cache = empty_cache(this.config.phonon.flow_lm);
        try {
            const outputs = await this._run_flow(cache, { voice, ...this.default_conditions });
            outputs.latent_out.dispose();
            outputs.eos_logit.dispose();
            return next_state(outputs);
        } finally {
            dispose_state(cache);
        }
    }

    /**
     * Generate speech for prepared text chunks as 24 kHz PCM, one waveform per chunk.
     *
     * The voice prefix is conditioned once per call, shared across all chunks, and freed when the
     * generator finishes, throws, or is exited early. Nothing is cached between calls, and
     * `speaker_embeddings` remains owned by the caller.
     *
     * @param {Object} options Generation inputs and conditioning controls.
     * @param {{ input_ids: number[], frames_after_eos: number }[]} options.chunks Text chunks to speak, in order,
     * as returned by `PocketTTSProcessor`. `frames_after_eos` is the number of frames generated after end of speech.
     * @param {Tensor} options.speaker_embeddings Voice latents of shape `[1, voice_dim, frames]`.
     * @param {number} [options.temperature] Flow noise variance (as in Python transformers); 0 = deterministic.
     * @param {number} [options.num_speakers] Speaker conditioning index (`0 | 1 | 2`), not a count.
     * @param {number} [options.duration_delta] Duration conditioning value.
     * @returns {AsyncGenerator<Float32Array>} The PCM of each chunk, in order.
     *
     * **Example:** Play each text chunk (one or more sentences) as soon as it is ready. Voices are flat float32 files in `[1, 512, frames]` order.
     * ```javascript
     * import { AutoProcessor, GradiumPhononTTSForConditionalGeneration, Tensor } from '@huggingface/transformers';
     * const id = 'onnx-community/phonon-31m-en-ONNX';
     * const processor = await AutoProcessor.from_pretrained(id);
     * const model = await GradiumPhononTTSForConditionalGeneration.from_pretrained(id, { dtype: 'q4' });
     * const voice = await fetch(`https://huggingface.co/${id}/resolve/main/voices/default.bin`);
     * const data = new Float32Array(await voice.arrayBuffer());
     * const speaker_embeddings = new Tensor('float32', data, [1, 512, data.length / 512]);
     * const chunks = await processor('Hello from Phonon. This is the second sentence.');
     * for await (const pcm of model.generate_speech({ chunks, speaker_embeddings })) {
     *   console.log(pcm.length); // mono samples at 24 kHz, one text chunk at a time
     * }
     * ```
     */
    async *generate_speech({
        chunks,
        speaker_embeddings,
        // @ts-expect-error TS2339
        temperature = this.config.phonon.temperature,
        num_speakers = this.default_conditions.num_speakers,
        duration_delta = this.default_conditions.duration_delta,
    }) {
        const conditions = { num_speakers, duration_delta };
        const prefix = await this._prepare_voice(speaker_embeddings);
        try {
            for (const { input_ids, frames_after_eos } of chunks) {
                const latents = await this._generate_latents(
                    input_ids,
                    frames_after_eos,
                    prefix,
                    temperature,
                    conditions,
                );
                yield await this._decode(latents);
            }
        } finally {
            dispose_state(prefix);
        }
    }

    // Runs the flow model for one chunk; returns its latents as `[1, frames, latent_dim]`.
    async _generate_latents(input_ids, frames_after_eos, prefix, temperature, conditions) {
        // @ts-expect-error TS2339
        const cfg = this.config.phonon;
        const latents = [];
        let state = null,
            latent = null;
        try {
            const prompt = await this._run_flow(prefix, { text: input_ids, ...conditions });
            state = next_state(prompt);
            prompt.latent_out.dispose();
            prompt.eos_logit.dispose();
            latent = full([1, 1, cfg.latent_dim], NaN);
            let countdown = null;
            // Frame budget of the original runtime; the EOS countdown follows Python's PocketTTSGenerationMixin.
            const max_frames = Math.ceil((input_ids.length / 3 + 2) * cfg.frame_rate);
            for (let frame = 0; frame < max_frames; ++frame) {
                const noise =
                    temperature === 0
                        ? full([1, cfg.latent_dim], 0)
                        : randn([1, cfg.latent_dim]).mul_(Math.sqrt(temperature));
                const step = await this._run_flow(state, { latent, noise, ...conditions });
                dispose_state(state);
                state = next_state(step);
                latent.dispose();
                latent = step.latent_out;
                latents.push(new Float32Array(latent.data));
                const eos = step.eos_logit.data[0];
                step.eos_logit.dispose();
                if (countdown === null && eos > cfg.eos_threshold) countdown = frames_after_eos;
                if (countdown !== null && countdown-- === 0) break;
            }
            if (countdown === null) {
                logger.warn('Phonon reached its safety frame limit before completing speech; audio may be truncated.');
            }
        } finally {
            dispose_state(state);
            latent?.dispose();
        }
        const data = new Float32Array(latents.length * cfg.latent_dim);
        latents.forEach((values, i) => data.set(values, i * cfg.latent_dim));
        return new Tensor('float32', data, [1, latents.length, cfg.latent_dim]);
    }

    // Decodes a whole chunk at once: one decoder call from an empty streaming state.
    async _decode(latents) {
        // @ts-expect-error TS2339
        const { mimi_decoder } = this.config.phonon;
        const inputs = { latent: latents, past_offset: i64([0], [1]), ...empty_cache(mimi_decoder) };
        for (const [name, dims] of Object.entries(mimi_decoder.conv_states)) inputs[name] = full(dims, 0);
        try {
            const outputs = await sessionRun(this.sessions.mimi_decoder, inputs);
            const pcm = new Float32Array(outputs.audio.data);
            dispose_state(outputs);
            return pcm;
        } finally {
            dispose_state(inputs);
        }
    }
}
