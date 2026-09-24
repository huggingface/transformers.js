import { PreTrainedModel } from '../modeling_utils.js';
import { ModelOutput } from '../modeling_outputs.js';
import { sessionRun } from '../session.js';
import { Tensor, cat } from '../../utils/tensor.js';

const LOG_HALF = Math.log(0.5);

/**
 * @param {number} x
 * @returns {number}
 */
const sigmoid = (x) => 1 / (1 + Math.exp(-x));

/**
 * Indices of the `k` largest values of `values` above `-Infinity`, in increasing order. Among equal values, the lowest
 * indices are kept.
 * @param {Float64Array} values
 * @param {number} k
 * @returns {number[]}
 */
function topk_indices(values, k) {
    // Most scores are `-Infinity` (the frames of silent speakers): only the others compete
    const candidates = [];
    for (let i = 0; i < values.length; ++i) {
        if (values[i] !== -Infinity) candidates.push(i);
    }
    if (candidates.length <= k) {
        return candidates;
    }
    // The k-th largest value is the threshold of the selection: every value above it is kept, and of the values equal
    // to it, as many as the k largest values count
    const sorted = Float64Array.from(candidates, (i) => values[i]).sort();
    const start = sorted.length - k;
    const threshold = sorted[start];
    let num_ties = 0;
    for (let i = start; i < sorted.length && sorted[i] === threshold; ++i) {
        ++num_ties;
    }
    return candidates.filter((i) => values[i] > threshold || (values[i] === threshold && num_ties-- > 0));
}

/**
 * Streaming state of {@link Nemotron3DiarizationForAudioFrameClassification}: the Arrival-Order Speaker Cache and the
 * FIFO queue of the most recent encoder frames, that every chunk attends to.
 */
export class Nemotron3DiarizationSpeakerCache {
    /**
     * @param {Object} config The streaming config of the model (`config.streaming_config`).
     * @param {Object} [options]
     * @param {number} [options.fifo_length] Capacity of the FIFO queue of the most recent encoder frames.
     * Defaults to `config.fifo_length`.
     * @param {number} [options.speaker_cache_update_period] Number of encoder frames moved from the FIFO queue to the
     * speaker cache when the queue overflows. Defaults to `config.speaker_cache_update_period`.
     */
    constructor(config, { fifo_length = undefined, speaker_cache_update_period = undefined } = {}) {
        this.fifo_length = fifo_length ?? config.fifo_length;
        this.speaker_cache_update_period = speaker_cache_update_period ?? config.speaker_cache_update_period;
        this.speaker_cache_length = config.speaker_cache_length;
        this.num_silence_frames = config.speaker_cache_silence_frames_per_speaker;
        this.prediction_score_threshold = config.prediction_score_threshold;
        this.latest_frames_score_boost = config.latest_frames_score_boost;
        this.num_speakers = config.num_speakers;
        this.subsampling_factor = config.subsampling_factor;

        // share of the speaker cache every speaker is budgeted, excluding its reserved silence slots, and the frame
        // counts the score policy spends it on when the cache is compressed
        const budget = Math.floor(this.speaker_cache_length / this.num_speakers) - this.num_silence_frames;
        this.min_positive_scores = Math.floor(budget * config.min_positive_scores_rate);
        this.num_strong_boosted_frames = Math.floor(budget * config.strong_boost_rate);
        this.num_weak_boosted_frames = Math.floor(budget * config.weak_boost_rate);

        /**
         * @type {Float32Array} The speaker cache frames, then the FIFO queue frames, of shape
         * `[batch_size, capacity, hidden_size]`: contiguous, so that they are the cached frames of the next step as they
         * are, and that frames move from the FIFO queue to the speaker cache without being copied.
         */
        this.frames = null;
        /** @type {Float32Array} Speaker probabilities of the cache frames, `[batch_size, speaker_cache_length, num_speakers]`. */
        this.probs = null;
        /** @type {Float32Array} Scratch buffer of the frames kept by a compression, `[speaker_cache_length, hidden_size]`. */
        this._kept_frames = null;
        this.batch_size = 0;
        this.hidden_size = 0;
        this.capacity = 0;
        this.num_cache_frames = 0;
        this.num_fifo_frames = 0;
        this.is_compressed = false;
        this.is_initialized = false;
    }

    /**
     * @param {number} batch_size
     * @param {number} hidden_size
     */
    lazy_initialization(batch_size, hidden_size) {
        this.batch_size = batch_size;
        this.hidden_size = hidden_size;
        this.probs = new Float32Array(batch_size * this.speaker_cache_length * this.num_speakers);
        this._kept_frames = new Float32Array(this.speaker_cache_length * hidden_size);
        this._reserve(this.speaker_cache_length + this.fifo_length);
        this.is_initialized = true;
    }

    /**
     * Grows the frame buffer to hold at least `capacity` frames per sample.
     * @param {number} capacity
     * @private
     */
    _reserve(capacity) {
        if (capacity <= this.capacity) {
            return;
        }
        const { batch_size, hidden_size } = this;
        const frames = new Float32Array(batch_size * capacity * hidden_size);
        if (this.frames) {
            const length = (this.num_cache_frames + this.num_fifo_frames) * hidden_size;
            const stride = this.capacity * hidden_size;
            for (let b = 0; b < batch_size; ++b) {
                frames.set(this.frames.subarray(b * stride, b * stride + length), b * capacity * hidden_size);
            }
        }
        this.frames = frames;
        this.capacity = capacity;
    }

    /**
     * The frames every chunk attends to: the speaker cache frames, then the FIFO queue frames. For a single sample, a
     * view of the state (no copy), that the next `update` overwrites.
     * @param {number} batch_size
     * @param {number} hidden_size
     * @returns {Tensor} The cached frames, of shape `[batch_size, num_cache_frames + num_fifo_frames, hidden_size]`.
     */
    get_embeds(batch_size, hidden_size) {
        if (!this.is_initialized) {
            this.lazy_initialization(batch_size, hidden_size);
        }
        const num_frames = this.num_cache_frames + this.num_fifo_frames;
        const length = num_frames * hidden_size;
        let data;
        if (batch_size === 1) {
            data = this.frames.subarray(0, length);
        } else {
            data = new Float32Array(batch_size * length);
            const stride = this.capacity * hidden_size;
            for (let b = 0; b < batch_size; ++b) {
                data.set(this.frames.subarray(b * stride, b * stride + length), b * length);
            }
        }
        return new Tensor('float32', data, [batch_size, num_frames, hidden_size]);
    }

    /**
     * No frames move to the speaker cache until the FIFO overflows, then at least `speaker_cache_update_period`
     * oldest frames are moved, plus any additional frames needed to restore fifo capacity.
     * @param {number} num_fifo_frames
     * @private
     */
    _num_popped_frames(num_fifo_frames) {
        if (num_fifo_frames <= this.fifo_length) {
            return 0;
        }
        const num_popped = Math.max(this.speaker_cache_update_period, num_fifo_frames - this.fifo_length);
        return Math.min(num_popped, num_fifo_frames);
    }

    /**
     * Pushes a processed chunk to the FIFO queue, moving its oldest frames to the speaker cache when it overflows.
     *
     * @param {Tensor} chunk_embeds Encoder frames of the chunk (and its look-ahead), of shape
     * `[batch_size, num_frames, hidden_size]`. Only the first `num_chunk_frames` join the FIFO queue: the look-ahead
     * frames after them are fed again at the next step.
     * @param {Tensor} chunk_logits Speaker logits of the step (cached frames, then chunk frames), of shape
     * `[batch_size, num_input_frames * subsampling_factor, num_speakers]`, used to score the frames when the speaker
     * cache is compressed.
     * @param {Tensor} silence_embeds Learned silence embedding of shape `[hidden_size]`, filling the reserved silence
     * slots of a compressed cache.
     * @param {number} num_chunk_frames Number of chunk frames that join the FIFO queue.
     * @param {BigInt64Array|null} [mask=null] Valid frames of the step (cached frames, then chunk frames), of shape
     * `[batch_size, num_input_frames]`, whose padding frames are given zero speaker probabilities.
     */
    update(chunk_embeds, chunk_logits, silence_embeds, num_chunk_frames, mask = null) {
        const [batch_size, num_embeds, hidden_size] = chunk_embeds.dims;
        if (!this.is_initialized) {
            this.lazy_initialization(batch_size, hidden_size);
        }
        const { num_cache_frames, num_fifo_frames, num_speakers, speaker_cache_length } = this;
        const num_queued_frames = num_fifo_frames + num_chunk_frames;
        const num_popped = this._num_popped_frames(num_queued_frames);
        // The popped frames are the oldest queued frames, right after the cache frames: they join them as they are,
        // compressed if they overflow the cache
        const num_frames = num_cache_frames + num_popped;
        const compress = num_frames > speaker_cache_length;

        // The chunk frames join the FIFO queue
        this._reserve(num_cache_frames + num_queued_frames);
        const stride = this.capacity * hidden_size;
        const chunk_data = /** @type {Float32Array} */ (chunk_embeds.data);
        for (let b = 0; b < batch_size; ++b) {
            const chunk_offset = b * num_embeds * hidden_size;
            this.frames.set(
                chunk_data.subarray(chunk_offset, chunk_offset + num_chunk_frames * hidden_size),
                b * stride + (num_cache_frames + num_fifo_frames) * hidden_size,
            );
        }

        if (num_popped > 0) {
            const num_logit_frames = chunk_logits.dims[1];
            const logits_data = /** @type {Float32Array} */ (chunk_logits.data);
            const step_length = mask ? mask.length / batch_size : 0;
            for (let b = 0; b < batch_size; ++b) {
                const frames = this.frames.subarray(b * stride, (b + 1) * stride);
                const cache_probs = this.probs.subarray(
                    b * speaker_cache_length * num_speakers,
                    (b + 1) * speaker_cache_length * num_speakers,
                );

                // The step (cache frames, then queued frames) estimates their speaker probabilities, except for the
                // frames of a compressed cache: out of order, their stored probabilities are the only ones
                let probs = this._pool_probs(
                    logits_data.subarray(
                        b * num_logit_frames * num_speakers,
                        (b + 1) * num_logit_frames * num_speakers,
                    ),
                    mask?.subarray(b * step_length, (b + 1) * step_length),
                    this.is_compressed ? num_cache_frames : 0,
                    num_frames,
                );
                if (this.is_compressed) {
                    probs.set(cache_probs.subarray(0, num_cache_frames * num_speakers));
                }

                if (compress) {
                    probs = this._compress(frames, probs, num_frames, silence_embeds);
                    // the FIFO queue follows the compressed cache
                    frames.copyWithin(
                        speaker_cache_length * hidden_size,
                        num_frames * hidden_size,
                        (num_cache_frames + num_queued_frames) * hidden_size,
                    );
                }
                cache_probs.set(probs);
            }
        }
        this.num_cache_frames = Math.min(num_frames, speaker_cache_length);
        this.num_fifo_frames = num_queued_frames - num_popped;
        this.is_compressed ||= compress;
    }

    /**
     * Speaker probabilities of the encoder frames `[start, end)` of one sample, zeroed on its padding frames.
     * @param {Float32Array} logits Speaker logits of the sample at the mel frame rate, of shape
     * `[num_frames * subsampling_factor, num_speakers]`.
     * @param {BigInt64Array|undefined} mask Valid encoder frames of the sample, of shape `[num_frames]`.
     * @param {number} start
     * @param {number} end
     * @returns {Float32Array} Probabilities of shape `[end, num_speakers]`, zero before `start`.
     * @private
     */
    _pool_probs(logits, mask, start, end) {
        const { subsampling_factor: factor, num_speakers } = this;
        const probs = new Float32Array(end * num_speakers);
        for (let t = start; t < end; ++t) {
            if (mask && !mask[t]) continue;
            for (let s = 0; s < num_speakers; ++s) {
                let sum = 0;
                for (let k = 0; k < factor; ++k) {
                    sum += sigmoid(logits[(t * factor + k) * num_speakers + s]);
                }
                probs[t * num_speakers + s] = sum / factor;
            }
        }
        return probs;
    }

    /**
     * Scores every (speaker, frame) pair of one sample: how well the frame represents the speaker alone. The frames
     * where the speaker is silent (and, for a speaker with enough positively scored frames, those where it overlaps
     * with others) score `-Infinity`.
     * @param {Float32Array} probs Speaker probabilities of the frames, of shape `[num_frames, num_speakers]`.
     * @param {number} num_frames
     * @param {number} num_scored_frames Number of frames per speaker of the scores, at least `num_frames`.
     * @returns {Float64Array} Scores of shape `[num_speakers, num_scored_frames]`, zero beyond `num_frames`.
     * @private
     */
    _get_frame_scores(probs, num_frames, num_scored_frames) {
        const { num_speakers, prediction_score_threshold: threshold } = this;
        const scores = new Float64Array(num_speakers * num_scored_frames);
        const log_complements = new Float64Array(num_speakers);
        for (let t = 0; t < num_frames; ++t) {
            const frame_probs = probs.subarray(t * num_speakers, (t + 1) * num_speakers);
            let log_complements_sum = 0;
            for (let s = 0; s < num_speakers; ++s) {
                log_complements[s] = Math.log(Math.max(1 - frame_probs[s], threshold));
                log_complements_sum += log_complements[s];
            }
            for (let s = 0; s < num_speakers; ++s) {
                const p = frame_probs[s];
                scores[s * num_scored_frames + t] =
                    p > 0.5
                        ? Math.log(Math.max(p, threshold)) - log_complements[s] + log_complements_sum - LOG_HALF
                        : -Infinity;
            }
        }
        for (let s = 0; s < num_speakers; ++s) {
            const speaker_scores = scores.subarray(s * num_scored_frames, s * num_scored_frames + num_frames);
            let num_positive = 0;
            for (const score of speaker_scores) {
                if (score > 0) ++num_positive;
            }
            if (num_positive >= this.min_positive_scores) {
                for (let t = 0; t < num_frames; ++t) {
                    if (!(speaker_scores[t] > 0)) speaker_scores[t] = -Infinity;
                }
            }
        }
        return scores;
    }

    /**
     * Keeps the `speaker_cache_length` most important frames of one sample, grouped by speaker and in their original
     * order within a speaker. `speaker_cache_silence_frames_per_speaker` slots per speaker are filled with
     * `silence_embeds`.
     * @param {Float32Array} frames Frame buffer of the sample, whose first `num_frames` frames (`[num_frames, hidden_size]`)
     * are replaced by the `speaker_cache_length` kept frames.
     * @param {Float32Array} probs Speaker probabilities of the frames, of shape `[num_frames, num_speakers]`.
     * @param {number} num_frames
     * @param {Tensor} silence_embeds Learned silence embedding of shape `[hidden_size]`.
     * @returns {Float32Array} The speaker probabilities of the kept frames.
     * @private
     */
    _compress(frames, probs, num_frames, silence_embeds) {
        const { hidden_size, num_speakers, speaker_cache_length } = this;
        const num_scored_frames = num_frames + this.num_silence_frames;
        const scores = this._get_frame_scores(probs, num_frames, num_scored_frames);
        const boosts = [
            [this.num_strong_boosted_frames, -2 * LOG_HALF],
            [this.num_weak_boosted_frames, -LOG_HALF],
        ];
        for (let s = 0; s < num_speakers; ++s) {
            const offset = s * num_scored_frames;
            const speaker_scores = scores.subarray(offset, offset + num_frames);
            // the frames beyond the cache capacity are the ones popped from the FIFO queue
            for (let t = speaker_cache_length; t < num_frames; ++t) {
                speaker_scores[t] += this.latest_frames_score_boost;
            }
            for (const [num_boosted, boost] of boosts) {
                for (const t of topk_indices(speaker_scores, num_boosted)) {
                    speaker_scores[t] += boost;
                }
            }
            // the silence frames are always kept
            scores.fill(Infinity, offset + num_frames, offset + num_scored_frames);
        }

        // The best (speaker, frame) pairs, in speaker then frame order. The slots left when fewer than
        // `speaker_cache_length` pairs score above `-Infinity` hold the silence embedding too.
        const kept = topk_indices(scores, speaker_cache_length);
        const silence_data = /** @type {Float32Array} */ (silence_embeds.data);
        const kept_frames = this._kept_frames;
        const kept_probs = new Float32Array(speaker_cache_length * num_speakers);
        for (let i = 0; i < speaker_cache_length; ++i) {
            const t = i < kept.length ? kept[i] % num_scored_frames : num_frames;
            if (t >= num_frames) {
                kept_frames.set(silence_data, i * hidden_size);
            } else {
                kept_frames.set(frames.subarray(t * hidden_size, (t + 1) * hidden_size), i * hidden_size);
                kept_probs.set(probs.subarray(t * num_speakers, (t + 1) * num_speakers), i * num_speakers);
            }
        }
        frames.set(kept_frames);
        return kept_probs;
    }
}

/**
 * Output of {@link Nemotron3DiarizationForAudioFrameClassification}.
 */
export class Nemotron3DiarizationOutput extends ModelOutput {
    /**
     * @param {Object} output The output of the model.
     * @param {Tensor} output.logits Per-frame speaker activity logits at the spectrogram frame rate, of shape
     * `[batch_size, num_frames, num_speakers]`. Their sigmoid gives the probability that each speaker is active in each
     * frame; speakers are ordered by their first arrival in the audio.
     * @param {Nemotron3DiarizationSpeakerCache|null} [output.speaker_cache] Updated streaming state (streaming mode
     * only), to pass to the forward of the next audio chunk of the same streams.
     */
    constructor({ logits, speaker_cache = null }) {
        super();
        this.logits = logits;
        this.speaker_cache = speaker_cache;
    }
}

export class Nemotron3DiarizationPreTrainedModel extends PreTrainedModel {}

/**
 * Streaming Sortformer speaker diarization model: predicts, for every spectrogram frame, the activity of up to
 * `config.head_config.num_speakers` speakers ordered by first arrival. Audio is processed chunk by chunk, each chunk
 * attending to a few look-ahead frames and to the Arrival-Order Speaker Cache and FIFO queue carried in a
 * {@link Nemotron3DiarizationSpeakerCache}. A whole recording is chunked by the forward itself (offline mode); a
 * stream is fed one chunk per forward (streaming mode).
 *
 * **Example:** Offline speaker diarization of a whole recording.
 *
 * ```javascript
 * import { AutoProcessor, AutoModelForAudioFrameClassification, load_audio } from '@huggingface/transformers';
 *
 * const model_id = 'onnx-community/Nemotron-3-Diarization-ONNX';
 * const processor = await AutoProcessor.from_pretrained(model_id);
 * const model = await AutoModelForAudioFrameClassification.from_pretrained(model_id);
 *
 * const url = 'https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/diarization_example.mp3';
 * const audio = await load_audio(url, processor.feature_extractor.config.sampling_rate);
 *
 * const inputs = await processor(audio);
 * const { logits } = await model(inputs); // [1, num_frames, 8], one frame every 10 ms
 * const segments = processor.extract_speaker_dict(logits, inputs.attention_mask)[0];
 * // [{ Start: 0.36, End: 9.31, Speaker: 0 }, { Start: 9.17, End: 13.83, Speaker: 1 }, ...]
 * ```
 *
 * **Example:** Streaming speaker diarization, one audio chunk at a time.
 *
 * ```javascript
 * processor.set_streaming_mode('low_latency'); // or 'very_low_latency', 'ultra_low_latency'
 *
 * async function* inputs_generator() {
 *   yield processor(audio.subarray(0, processor.num_samples_first_audio_chunk), { is_streaming: true });
 *   let mel_frame_idx = processor.num_mel_frames_per_step;
 *   let start_idx = processor.audio_chunk_start(mel_frame_idx);
 *   while (start_idx + processor.num_samples_per_audio_chunk <= audio.length) {
 *     const end_idx = start_idx + processor.num_samples_per_audio_chunk;
 *     yield processor(audio.subarray(start_idx, end_idx), { is_streaming: true, is_first_audio_chunk: false });
 *     mel_frame_idx += processor.num_mel_frames_per_step;
 *     start_idx = processor.audio_chunk_start(mel_frame_idx);
 *   }
 *   // the audio ended: the frames left in the buffer are the last ones of the session
 *   yield processor(audio.subarray(start_idx), { is_streaming: true, is_first_audio_chunk: false, is_last_audio_chunk: true });
 * }
 *
 * let speaker_cache = null;
 * const logits = [];
 * for await (const inputs of inputs_generator()) {
 *   const outputs = await model({ ...inputs, speaker_cache });
 *   logits.push(outputs.logits); // the chunk's frames, without its look-ahead
 *   speaker_cache = outputs.speaker_cache;
 * }
 * const segments = processor.extract_speaker_dict(cat(logits, 1))[0];
 * ```
 */
export class Nemotron3DiarizationForAudioFrameClassification extends Nemotron3DiarizationPreTrainedModel {
    /**
     * Runs the model on a whole recording (offline mode) or on one chunk of a streaming session (streaming mode).
     *
     * The two optional arguments select the mode. Streaming mode, one chunk per forward: `num_lookahead_frames` given
     * (a first chunk creates the `speaker_cache`, later chunks receive it), or `speaker_cache` given alone (the last
     * chunk of the session, no look-ahead). The input minus its look-ahead is one chunk, whatever its length, pushed
     * as a whole to the FIFO queue sized by `config.streaming_config`. Offline mode, neither given: the input is a
     * whole recording, split into chunks of `config.chunk_length` encoder frames that take up to
     * `config.chunk_right_context` look-ahead frames from the following ones, with a FIFO queue sized by
     * `config.fifo_length`; no cache is returned.
     *
     * @param {Object} model_inputs
     * @param {Tensor} model_inputs.input_features Log-Mel features of shape `[batch_size, num_frames, num_mel_bins]`.
     * @param {Tensor} [model_inputs.attention_mask=null] Valid frames of each sample of a padded batch, of shape
     * `[batch_size, num_frames]`.
     * @param {Nemotron3DiarizationSpeakerCache} [model_inputs.speaker_cache=null] Streaming state returned by the
     * forward of the previous chunk of the same audio streams.
     * @param {number} [model_inputs.num_lookahead_frames=null] Streaming mode: number of trailing encoder frames of the
     * input that are look-ahead only. They are attended to, but their logits are not returned and they do not join
     * the FIFO queue, as they open the next chunk. The processor sets it for every chunk but the last one of a session.
     * @returns {Promise<Nemotron3DiarizationOutput>}
     */
    async forward({ input_features, attention_mask = null, speaker_cache = null, num_lookahead_frames = null }) {
        const config = /** @type {any} */ (this.config);
        const is_streaming = num_lookahead_frames !== null || speaker_cache !== null;
        if (speaker_cache === null) {
            // the cache defaults to the streaming FIFO sizes, offline mode overrides them
            const { fifo_length, speaker_cache_update_period } = config;
            const offline_sizes = is_streaming ? {} : { fifo_length, speaker_cache_update_period };
            speaker_cache = new Nemotron3DiarizationSpeakerCache(config.streaming_config, offline_sizes);
        }
        num_lookahead_frames ??= 0;

        const factor = config.audio_config.subsampling_factor;
        const hidden_size = config.audio_config.hidden_size;
        const [batch_size, num_frames] = input_features.dims;
        const num_embeds = Math.ceil(num_frames / factor);
        const num_chunk_embeds = num_embeds - num_lookahead_frames;
        if (num_lookahead_frames < 0 || num_chunk_embeds < 1) {
            throw new Error(
                `\`num_lookahead_frames\` (${num_lookahead_frames}) must be between 0 and one less than the number of ` +
                    `encoder frames of the input (${num_embeds}).`,
            );
        }

        // Valid encoder frames: the first mel frame of each group of `subsampling_factor`
        const embed_mask = new BigInt64Array(batch_size * num_embeds).fill(1n);
        if (attention_mask) {
            const mask_data = attention_mask.data;
            for (let b = 0; b < batch_size; ++b) {
                for (let t = 0; t < num_embeds; ++t) {
                    embed_mask[b * num_embeds + t] = Number(mask_data[b * num_frames + t * factor]) ? 1n : 0n;
                }
            }
        }

        const [chunk_length, chunk_right_context] = is_streaming
            ? [num_chunk_embeds, num_lookahead_frames]
            : [config.chunk_length, config.chunk_right_context];

        const session = this.sessions['model'];
        const logits = [];
        for (let start_idx = 0; start_idx < num_chunk_embeds; start_idx += chunk_length) {
            const end_idx = Math.min(start_idx + chunk_length, num_chunk_embeds);
            const num_chunk_frames = end_idx - start_idx;
            const stop_idx = Math.min(end_idx + chunk_right_context, num_embeds);
            const num_step_embeds = stop_idx - start_idx;

            const cached_embeds = speaker_cache.get_embeds(batch_size, hidden_size);
            const cached_length = cached_embeds.dims[1];

            // Mask of the step: the cached frames are always valid
            const step_length = cached_length + num_step_embeds;
            const step_mask = new BigInt64Array(batch_size * step_length).fill(1n);
            for (let b = 0; b < batch_size; ++b) {
                step_mask.set(
                    embed_mask.subarray(b * num_embeds + start_idx, b * num_embeds + stop_idx),
                    b * step_length + cached_length,
                );
            }

            // A streaming step covers the whole input
            const step_start = start_idx * factor;
            const step_end = Math.min(stop_idx * factor, num_frames);
            const step_features =
                step_start === 0 && step_end === num_frames
                    ? input_features
                    : input_features.slice(null, [step_start, step_end], null);

            const outputs = await sessionRun(session, {
                input_features: step_features,
                cached_embeds,
                attention_mask: new Tensor('int64', step_mask, [batch_size, step_length]),
            });
            speaker_cache.update(
                outputs.chunk_embeds,
                outputs.logits,
                outputs.silence_embeds,
                num_chunk_frames,
                step_mask,
            );

            // with no look-ahead, the last encoder frame may be the padding added by feature stacking
            const num_chunk_logits = Math.min(num_chunk_frames * factor, num_frames - step_start);
            const logits_start = cached_length * factor;
            logits.push(outputs.logits.slice(null, [logits_start, logits_start + num_chunk_logits], null));
        }

        return new Nemotron3DiarizationOutput({
            logits: logits.length === 1 ? logits[0] : cat(logits, 1),
            speaker_cache: is_streaming ? speaker_cache : null,
        });
    }
}
