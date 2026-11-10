/**
 * @module models
 */
import { PreTrainedModel, addPastKeyValues, getPastKeyValues } from '../modeling_utils.js';
import { Tensor, full, randn } from '../../utils/tensor.js';
import { sessionRun } from '../session.js';
import { logger } from '../../utils/logger.js';

/** Base class for Gradium Phonon TTS models. */
export class GradiumPhononTTSPreTrainedModel extends PreTrainedModel {}

/** Gradium Phonon speech generation from text and precomputed voice latents. */
export class GradiumPhononTTSForConditionalGeneration extends GradiumPhononTTSPreTrainedModel {
    /**
     * Generate a waveform for one prompt prepared by `PocketTTSProcessor`.
     * Runs both the speech generator and the Mimi decoder, trimming unused frames before decoding.
     * The exported graph fixes the block size and number of flow-solver steps.
     *
     * @param {Object} options Processor outputs and generation options.
     * @param {Tensor} options.input_ids Prompt tokens, including audio placeholders, of shape `[1, sequence_length]`.
     * @param {Tensor} [options.attention_mask] Mask of non-padding prompt tokens.
     * @param {Tensor} [options.audio_prompt_latents] Voice latents of shape `[1, frames, audio_prompt_dim]`.
     * @param {Tensor} [options.audio_prompt_latents_mask] Mask of non-padding voice frames.
     * @param {Tensor|number} [options.frames_after_eos=1] Frames to retain after the EOS frame.
     * @param {Object} [options.generation_config] Generation defaults to override.
     * @param {number} [options.max_new_tokens] Maximum number of audio frames.
     * @param {number} [options.max_length] Maximum prompt-plus-audio sequence length.
     * @param {number} [options.temperature] Noise variance; zero is deterministic.
     * @param {boolean} [options.do_sample] Whether to sample Gaussian noise.
     * @param {number} [options.num_flow_steps] Solver steps per frame (fixed by the export).
     * @param {number} [options.eos_threshold] End-of-speech logit threshold.
     * @param {number} [options.min_frames_before_eos] Frames before EOS may be detected.
     * @returns {Promise<{audio_values: Tensor}>} Waveform of shape `[1, 1, samples]`. The caller owns the returned tensor.
     */
    async generate({
        input_ids,
        attention_mask = full(input_ids.dims, 1n),
        audio_prompt_latents = full([1, 0, /** @type {any} */ (this.config).audio_prompt_dim], 0),
        audio_prompt_latents_mask = full(audio_prompt_latents.dims.slice(0, 2), 1n),
        frames_after_eos = 1,
        generation_config,
        ...kwargs
    }) {
        const config = /** @type {any} */ (this.config);
        const options = {
            do_sample: true,
            temperature: 0.3,
            num_flow_steps: 1,
            eos_threshold: config.eos_threshold,
            min_frames_before_eos: 0,
            ...this.generation_config,
            ...generation_config,
            ...Object.fromEntries(Object.entries(kwargs).filter(([, value]) => value != null)),
        };
        const { model, mimi_decoder } = this.sessions;
        const block_size = model.inputMetadata.find(({ name }) => name === 'noise').shape[1];
        if (options.num_flow_steps !== 1) {
            throw new Error('This ONNX export supports num_flow_steps=1; other values require re-export.');
        }
        if (input_ids.dims.length !== 2 || input_ids.dims[0] !== 1) {
            throw new Error('This Phonon ONNX export supports one prompt per generate() call.');
        }
        const prompt_length = Number(attention_mask.sum().item());
        const text_length = prompt_length - Number(audio_prompt_latents_mask.sum().item()) - 1;
        const max_frames =
            options.max_new_tokens ??
            (options.max_length != null
                ? options.max_length - prompt_length
                : Math.ceil((text_length / 3 + 2) * config.frame_rate));
        const tail = frames_after_eos instanceof Tensor ? Number(frames_after_eos.item()) : frames_after_eos;
        if (!Number.isInteger(max_frames) || max_frames < 1 || !Number.isInteger(tail) || tail < 0) {
            throw new Error('The frame budget must be a positive integer and frames_after_eos a non-negative integer.');
        }
        if (!Number.isFinite(options.temperature) || options.temperature < 0) {
            throw new Error('temperature must be a finite non-negative number.');
        }
        let inputs = { input_ids, attention_mask, audio_prompt_latents, audio_prompt_latents_mask };
        const empty_inputs = {
            input_ids: new Tensor('int64', [], [1, 0]),
            attention_mask: new Tensor('int64', [], [1, 0]),
            audio_prompt_latents: full([1, 0, config.audio_prompt_dim], 0),
            audio_prompt_latents_mask: new Tensor('int64', [], [1, 0]),
        };
        const cache = addPastKeyValues(this, {}, null);
        let audio_latents,
            num_frames = 0,
            stop_frame = null;
        try {
            while (num_frames < Math.min(stop_frame ?? max_frames, max_frames)) {
                const noise =
                    options.do_sample && options.temperature > 0
                        ? randn([1, block_size, config.latent_dim]).mul_(Math.sqrt(options.temperature))
                        : full([1, block_size, config.latent_dim], 0);
                const outputs = await sessionRun(model, { ...inputs, ...cache, noise });
                getPastKeyValues(outputs, cache);
                audio_latents?.dispose();
                audio_latents = outputs.audio_latents;
                if (stop_frame === null) {
                    const eos = outputs.eos_logits.data.findIndex(
                        (value, i) => num_frames + i >= options.min_frames_before_eos && value > options.eos_threshold,
                    );
                    if (eos !== -1) stop_frame = num_frames + eos + tail + 1;
                }
                num_frames += block_size;
                inputs = empty_inputs;
            }
            num_frames = Math.min(num_frames, stop_frame ?? max_frames, max_frames);
            if (stop_frame === null && options.max_new_tokens == null && options.max_length == null) {
                logger.warn('Phonon reached its safety frame limit; audio may be truncated.');
            }
            const { audio_values } = await sessionRun(mimi_decoder, {
                audio_latents,
                num_frames: new Tensor('int64', [BigInt(num_frames)], []),
            });
            return { audio_values };
        } finally {
            await cache.dispose();
            audio_latents?.dispose();
        }
    }
}
