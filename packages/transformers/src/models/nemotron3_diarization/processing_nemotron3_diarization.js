import { AutoFeatureExtractor } from '../auto/feature_extraction_auto.js';
import { Processor } from '../../processing_utils.js';
import { Tensor } from '../../utils/tensor.js';
import { validate_audio_inputs } from '../../feature_extraction_utils.js';

/**
 * Streaming modes of the released checkpoint: name to `[chunk_length, chunk_right_context]` in encoder frames.
 */
const DEFAULT_STREAMING_MODES = Object.freeze({
    low_latency: [9, 4], // 1.04 s
    very_low_latency: [6, 2], // 0.64 s
    ultra_low_latency: [3, 1], // 0.32 s
});

/**
 * Rounds a number to the given number of decimals, like Python's `round`.
 * @param {number} x
 * @param {number} decimals
 */
function round(x, decimals) {
    return Number(x.toFixed(decimals));
}

/**
 * @typedef {Object} SpeakerSegment
 * @property {number} Start The start time of the segment, in seconds.
 * @property {number} End The end time of the segment, in seconds.
 * @property {number} Speaker The index of the speaker, speakers being numbered in order of first arrival.
 */

/**
 * Processor for Nemotron-3-Diarization: extracts the log-Mel features of a whole recording (offline) or of one
 * chunk of a streaming session, and turns the model's speaker logits into speech segments.
 */
export class Nemotron3DiarizationProcessor extends Processor {
    static feature_extractor_class = AutoFeatureExtractor;
    static uses_processor_config = true;

    /** @type {string} Streaming mode of the sessions, one of `streaming_modes`, set with `set_streaming_mode`. */
    streaming_mode;

    constructor(config, components, chat_template) {
        super(config, components, chat_template);

        /** Number of mel frames per encoder frame. */
        this.subsampling_factor = this.config.subsampling_factor ?? 8;

        /** Streaming modes the checkpoint supports, name to `[chunk_length, chunk_right_context]` in encoder frames. */
        this.streaming_modes = this.config.streaming_modes ?? DEFAULT_STREAMING_MODES;

        this.set_streaming_mode(this.config.streaming_mode ?? 'low_latency');
    }

    /**
     * Selects the streaming mode of the sessions among `streaming_modes`: every chunk size below and the
     * `num_lookahead_frames` given to the model re-derive from it.
     * @param {string} streaming_mode The streaming mode, e.g., `"low_latency"`, `"very_low_latency"` or `"ultra_low_latency"`.
     */
    set_streaming_mode(streaming_mode) {
        if (!Object.hasOwn(this.streaming_modes, streaming_mode)) {
            throw new Error(
                `Unknown \`streaming_mode\` "${streaming_mode}", expected one of ${JSON.stringify(Object.keys(this.streaming_modes))}.`,
            );
        }
        this.streaming_mode = streaming_mode;
    }

    /**
     * `[chunk_length, chunk_right_context]` of the streaming mode, in encoder frames.
     * @type {[number, number]}
     * @private
     */
    get _streaming_chunk_sizes() {
        return this.streaming_modes[this.streaming_mode];
    }

    /**
     * Input buffer latency (ms) of the streaming mode: the model emits a chunk once its look-ahead frames have
     * arrived, i.e. after `(chunk_length + chunk_right_context)` encoder frames.
     * @type {number}
     */
    get streaming_latency_ms() {
        const [chunk_length, chunk_right_context] = this._streaming_chunk_sizes;
        const { hop_length, sampling_rate } = this.feature_extractor.config;
        const encoder_frame_ms = ((this.subsampling_factor * hop_length) / sampling_rate) * 1000;
        return Math.round((chunk_length + chunk_right_context) * encoder_frame_ms);
    }

    /**
     * Number of mel frames each streaming chunk carries: its own frames plus the look-ahead frames.
     * @type {number}
     */
    get num_mel_frames_per_audio_chunk() {
        const [chunk_length, chunk_right_context] = this._streaming_chunk_sizes;
        return (chunk_length + chunk_right_context) * this.subsampling_factor;
    }

    /**
     * Number of mel frames the model emits per streaming chunk, i.e. how far the frame cursor advances between two
     * chunks (`num_mel_frames_per_audio_chunk` minus the look-ahead frames).
     * @type {number}
     */
    get num_mel_frames_per_step() {
        const [chunk_length] = this._streaming_chunk_sizes;
        return chunk_length * this.subsampling_factor;
    }

    /**
     * Number of audio samples to feed for the first chunk of a session (`is_first_audio_chunk=true`, centered
     * windows) so that the processor returns exactly `num_mel_frames_per_audio_chunk` frames.
     * @type {number}
     */
    get num_samples_first_audio_chunk() {
        const { hop_length, win_length } = this.feature_extractor.config;
        return (this.num_mel_frames_per_audio_chunk - 1) * hop_length + Math.floor(win_length / 2);
    }

    /**
     * Number of audio samples to feed for a later chunk of a session (`is_first_audio_chunk=false`, uncentered
     * windows) so that the processor returns exactly `num_mel_frames_per_audio_chunk` frames.
     * @type {number}
     */
    get num_samples_per_audio_chunk() {
        const { hop_length, win_length } = this.feature_extractor.config;
        return this.num_mel_frames_per_audio_chunk * hop_length + win_length;
    }

    /**
     * First audio sample of the chunk starting at `mel_frame_idx`. An uncentered window starts half a transform
     * before the frame it belongs to, so a chunk starts `n_fft / 2` samples before its first frame.
     * @param {number} mel_frame_idx The index of the first mel frame of the chunk.
     * @returns {number} The index of the first audio sample of the chunk.
     */
    audio_chunk_start(mel_frame_idx) {
        const { hop_length, n_fft } = this.feature_extractor.config;
        return mel_frame_idx * hop_length - Math.floor(n_fft / 2);
    }

    /**
     * Extracts the model inputs of a whole recording, or of one chunk of a streaming session.
     *
     * In streaming mode, the trailing frames whose analysis window reaches past the chunk are dropped, so
     * `input_features` holds exactly the frames of the chunk, and every chunk but the last also carries
     * `num_lookahead_frames`, the number of its trailing look-ahead encoder frames, which puts the model in
     * streaming mode.
     *
     * @param {Float32Array|Float64Array} audio The audio waveform, sampled at `feature_extractor.config.sampling_rate`.
     * @param {Object} [options]
     * @param {boolean} [options.is_streaming=false] Whether the audio is one chunk of a streaming session. Every chunk
     * but the last must hold exactly `num_samples_first_audio_chunk` audio samples for the first one and
     * `num_samples_per_audio_chunk` for the later ones.
     * @param {boolean} [options.is_first_audio_chunk=true] Whether this is the first chunk of a streaming session.
     * The analysis windows are centered for the first chunk and for offline use, and not for the later chunks, so that
     * the per-chunk spectrogram reproduces, frame for frame, a single full-utterance pass. Must be `true` when
     * `is_streaming=false`.
     * @param {boolean} [options.is_last_audio_chunk=false] Whether this chunk ends the streaming session. The last chunk
     * has no look-ahead, so every one of its frames is scored, whatever their number. Must be `false` when
     * `is_streaming=false`.
     * @returns {Promise<{ input_features: Tensor; attention_mask: Tensor; num_lookahead_frames?: number }>}
     */
    async _call(audio, { is_streaming = false, is_first_audio_chunk = true, is_last_audio_chunk = false } = {}) {
        validate_audio_inputs(audio, 'Nemotron3DiarizationProcessor');

        if (!is_streaming && (!is_first_audio_chunk || is_last_audio_chunk)) {
            throw new Error(
                'In non-streaming mode (`is_streaming=false`), `is_first_audio_chunk` must be `true` and ' +
                    '`is_last_audio_chunk` must be `false`.',
            );
        }

        /** @type {{ input_features: Tensor; attention_mask: Tensor; num_lookahead_frames?: number }} */
        const outputs = await this.feature_extractor(audio, { center: is_first_audio_chunk });
        if (!is_streaming) {
            return outputs;
        }

        // Drop the trailing frames whose analysis window reaches past the chunk (the valid frames come first)
        const mask_data = /** @type {BigInt64Array} */ (outputs.attention_mask.data);
        const first_padding_frame = mask_data.indexOf(0n);
        const num_frames = first_padding_frame === -1 ? mask_data.length : first_padding_frame;
        if (num_frames < mask_data.length) {
            outputs.input_features = outputs.input_features.slice(null, [0, num_frames], null);
            outputs.attention_mask = outputs.attention_mask.slice(null, [0, num_frames]);
        }
        if (!is_last_audio_chunk) {
            const expected_num_frames = this.num_mel_frames_per_audio_chunk;
            if (num_frames !== expected_num_frames) {
                const which = is_first_audio_chunk ? 'num_samples_first_audio_chunk' : 'num_samples_per_audio_chunk';
                throw new Error(
                    `A \`${this.streaming_mode}\` chunk must hold ${expected_num_frames} mel frames, got ${num_frames}: ` +
                        `feed \`${which}\` audio samples, or pass \`is_last_audio_chunk=true\` for the last chunk of the session.`,
                );
            }
            outputs.num_lookahead_frames = this._streaming_chunk_sizes[1];
        }
        return outputs;
    }

    /**
     * Turns the per-frame speaker logits of `Nemotron3DiarizationForAudioFrameClassification` into speech segments.
     *
     * @param {Tensor} logits Logits returned by the model, of shape `[batch_size, num_frames, num_speakers]`, at the
     * spectrogram frame rate (one per 10 ms). For a streaming session, the concatenated logits of its chunks.
     * @param {Tensor} [attention_mask=null] Valid frames of each sample of a padded batch, of shape `[batch_size, num_frames]`.
     * @param {number} [threshold=0.5] Speaker probability above which a frame counts as speech of that speaker.
     * @returns {SpeakerSegment[][]} For each sample, its speech segments sorted by start time. Overlapping speech gives
     * overlapping segments.
     */
    extract_speaker_dict(logits, attention_mask = null, threshold = 0.5) {
        const [batch_size, num_frames, num_speakers] = logits.dims;
        const { hop_length, sampling_rate } = this.feature_extractor.config;
        const frame_duration = hop_length / sampling_rate;
        const logits_data = /** @type {Float32Array} */ (logits.data);
        const mask_data = attention_mask?.data;

        const speaker_dicts = [];
        for (let b = 0; b < batch_size; ++b) {
            /** @type {SpeakerSegment[]} */
            const segments = [];
            for (let s = 0; s < num_speakers; ++s) {
                let start = -1;
                for (let t = 0; t <= num_frames; ++t) {
                    let active = false;
                    if (t < num_frames && (!mask_data || mask_data[b * num_frames + t])) {
                        const x = logits_data[(b * num_frames + t) * num_speakers + s];
                        // float32 sigmoid, as computed by the reference implementation
                        active = Math.fround(1 / (1 + Math.exp(-x))) > threshold;
                    }
                    if (active && start < 0) {
                        start = t;
                    } else if (!active && start >= 0) {
                        segments.push({
                            Start: round(start * frame_duration, 2),
                            End: round(t * frame_duration, 2),
                            Speaker: s,
                        });
                        start = -1;
                    }
                }
            }
            segments.sort((a, b) => a.Start - b.Start || a.Speaker - b.Speaker);
            speaker_dicts.push(segments);
        }
        return speaker_dicts;
    }
}
