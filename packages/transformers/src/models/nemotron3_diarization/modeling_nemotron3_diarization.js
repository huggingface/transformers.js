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
 * View of the `b`-th sample of a flat `[batch_size, ...]` buffer.
 * @template {Float32Array|BigInt64Array} T
 * @param {T} data
 * @param {number} b
 * @param {number} sample_size Number of elements per sample.
 * @returns {T}
 */
const sample_view = (data, b, sample_size) => /** @type {T} */ (data.subarray(b * sample_size, (b + 1) * sample_size));

/**
 * Streaming state of {@link Nemotron3DiarizationForAudioFrameClassification}: the Arrival-Order Speaker Cache and the
 * FIFO queue of the most recent encoder frames, that every chunk attends to.
 */
export class Nemotron3DiarizationSpeakerCache {
    /**
     * @param {Object} config Speaker-cache policy, and the FIFO sizes of streaming mode (`config.streaming_config`).
     * @param {Object} [options]
     * @param {number} [options.fifo_length] Capacity of the FIFO queue of the most recent encoder frames.
     * Defaults to `config.fifo_length`.
     * @param {number} [options.speaker_cache_update_period] Number of encoder frames moved from the FIFO queue to the
     * speaker cache when the queue overflows. Defaults to `config.speaker_cache_update_period`.
     */
    constructor(config, { fifo_length, speaker_cache_update_period } = {}) {
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

        /** @type {Float32Array} Speaker cache frames, of shape `[batch_size, speaker_cache_length, hidden_size]`. */
        this.embeds = null;
        /** @type {Float32Array} Speaker probabilities of the cache frames, of shape `[batch_size, speaker_cache_length, num_speakers]`. */
        this.probs = null;
        /** @type {Float32Array} FIFO queue frames, of shape `[batch_size, fifo_length, hidden_size]`. */
        this.fifo = null;
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
        this.embeds = new Float32Array(batch_size * this.speaker_cache_length * hidden_size);
        this.probs = new Float32Array(batch_size * this.speaker_cache_length * this.num_speakers);
        this.fifo = new Float32Array(batch_size * this.fifo_length * hidden_size);
        this.is_initialized = true;
    }

    /**
     * The frames every chunk attends to: the speaker cache frames, then the FIFO queue frames. Takes the sizes of the
     * chunk rather than the chunk, whose frames the model computes along with the logits.
     * @param {number} batch_size
     * @param {number} hidden_size
     * @returns {Tensor} The cached frames, of shape `[batch_size, num_cache_frames + num_fifo_frames, hidden_size]`.
     */
    get_embeds(batch_size, hidden_size) {
        if (!this.is_initialized) {
            this.lazy_initialization(batch_size, hidden_size);
        }
        const { num_cache_frames, num_fifo_frames } = this;
        const num_frames = num_cache_frames + num_fifo_frames;
        const data = new Float32Array(batch_size * num_frames * hidden_size);
        for (let b = 0; b < batch_size; ++b) {
            const embeds = sample_view(this.embeds, b, this.speaker_cache_length * hidden_size);
            const fifo = sample_view(this.fifo, b, this.fifo_length * hidden_size);
            const offset = b * num_frames * hidden_size;
            data.set(embeds.subarray(0, num_cache_frames * hidden_size), offset);
            data.set(fifo.subarray(0, num_fifo_frames * hidden_size), offset + num_cache_frames * hidden_size);
        }
        return new Tensor('float32', data, [batch_size, num_frames, hidden_size]);
    }

    /**
     * Speaker probabilities of one sample at the encoder frame rate, zeroed on the padding frames.
     * @param {Float32Array} logits Speaker logits at the mel frame rate, of shape
     * `[num_frames * subsampling_factor, num_speakers]`.
     * @param {BigInt64Array|null} mask Valid encoder frames, of shape `[num_frames]`.
     * @returns {Float32Array} Probabilities of shape `[num_frames, num_speakers]`.
     * @private
     */
    _pool_probs(logits, mask) {
        const { subsampling_factor: factor, num_speakers } = this;
        const num_frames = logits.length / (factor * num_speakers);
        const probs = new Float32Array(num_frames * num_speakers);
        for (let t = 0; t < num_frames; ++t) {
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
     * @param {Tensor} chunk_embeds Encoder input frames of the chunk and its look-ahead, of shape
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
        // the popped frames join the cache frames, compressed if they overflow the cache
        const num_frames = num_cache_frames + num_popped;
        const compress = num_frames > speaker_cache_length;

        const [, num_logit_frames] = chunk_logits.dims;
        const num_input_frames = num_logit_frames / this.subsampling_factor;
        const chunk_data = /** @type {Float32Array} */ (chunk_embeds.data);
        const logits_data = /** @type {Float32Array} */ (chunk_logits.data);
        for (let b = 0; b < batch_size; ++b) {
            const fifo = sample_view(this.fifo, b, this.fifo_length * hidden_size);
            const fifo_embeds = new Float32Array(num_queued_frames * hidden_size);
            fifo_embeds.set(fifo.subarray(0, num_fifo_frames * hidden_size));
            fifo_embeds.set(
                sample_view(chunk_data, b, num_embeds * hidden_size).subarray(0, num_chunk_frames * hidden_size),
                num_fifo_frames * hidden_size,
            );

            if (num_popped > 0) {
                const embeds = sample_view(this.embeds, b, speaker_cache_length * hidden_size);
                const stored_probs = sample_view(this.probs, b, speaker_cache_length * num_speakers);
                const probs = this._pool_probs(
                    sample_view(logits_data, b, num_logit_frames * num_speakers),
                    mask && sample_view(mask, b, num_input_frames),
                );

                /** @type {Float32Array} */
                let cache_embeds = new Float32Array(num_frames * hidden_size);
                cache_embeds.set(embeds.subarray(0, num_cache_frames * hidden_size));
                cache_embeds.set(fifo_embeds.subarray(0, num_popped * hidden_size), num_cache_frames * hidden_size);
                /** @type {Float32Array} */
                let cache_probs = probs.slice(0, num_frames * num_speakers);
                if (this.is_compressed) {
                    // an uncompressed cache still holds plain chunk frames, whose probabilities this step re-estimates;
                    // a compressed one is out of order, so the probs stored alongside its frames are the only ones
                    cache_probs.set(stored_probs.subarray(0, num_cache_frames * num_speakers));
                }

                if (compress) {
                    [cache_embeds, cache_probs] = this._compress(cache_embeds, cache_probs, silence_embeds);
                }
                embeds.set(cache_embeds);
                stored_probs.set(cache_probs);
            }
            fifo.set(fifo_embeds.subarray(num_popped * hidden_size));
        }
        this.num_cache_frames = Math.min(num_frames, speaker_cache_length);
        this.num_fifo_frames = num_queued_frames - num_popped;
        this.is_compressed ||= compress;
    }

    /**
     * Scores every (frame, speaker) pair of one sample: how well the frame represents the speaker alone. The frames
     * where the speaker is silent (and, for a speaker with enough positively scored frames, those where it overlaps
     * with others) score `-Infinity`.
     * @param {Float32Array} probs Speaker probabilities, of shape `[num_frames, num_speakers]`.
     * @returns {Float64Array} Scores of shape `[num_frames, num_speakers]`.
     * @private
     */
    _get_frame_scores(probs) {
        const { num_speakers, prediction_score_threshold: threshold } = this;
        const scores = new Float64Array(probs.length);
        const log_complements = new Float64Array(num_speakers);
        for (let offset = 0; offset < probs.length; offset += num_speakers) {
            let log_complements_sum = 0;
            for (let s = 0; s < num_speakers; ++s) {
                log_complements[s] = Math.log(Math.max(1 - probs[offset + s], threshold));
                log_complements_sum += log_complements[s];
            }
            for (let s = 0; s < num_speakers; ++s) {
                const p = probs[offset + s];
                scores[offset + s] =
                    p > 0.5
                        ? Math.log(Math.max(p, threshold)) - log_complements[s] + log_complements_sum - LOG_HALF
                        : -Infinity;
            }
        }
        for (let s = 0; s < num_speakers; ++s) {
            let num_positive = 0;
            for (let i = s; i < scores.length; i += num_speakers) {
                if (scores[i] > 0) ++num_positive;
            }
            if (num_positive >= this.min_positive_scores) {
                for (let i = s; i < scores.length; i += num_speakers) {
                    if (!(scores[i] > 0)) scores[i] = -Infinity;
                }
            }
        }
        return scores;
    }

    /**
     * Adds `boost` to the `num_boosted` best scores of every speaker.
     * @param {Float64Array} scores Scores of shape `[num_frames, num_speakers]`, updated in place.
     * @param {number} num_boosted
     * @param {number} boost
     * @private
     */
    _boost_scores(scores, num_boosted, boost) {
        const { num_speakers } = this;
        const speaker_scores = new Float64Array(scores.length / num_speakers);
        for (let s = 0; s < num_speakers; ++s) {
            for (let t = 0; t < speaker_scores.length; ++t) {
                speaker_scores[t] = scores[t * num_speakers + s];
            }
            for (const t of topk_indices(speaker_scores, num_boosted)) {
                scores[t * num_speakers + s] += boost;
            }
        }
    }

    /**
     * Keeps the `speaker_cache_length` most important frames of one sample, grouped by speaker and in their original
     * order within a speaker. `speaker_cache_silence_frames_per_speaker` slots per speaker are filled with
     * `silence_embeds`.
     * @param {Float32Array} embeds Frames of shape `[num_frames, hidden_size]`.
     * @param {Float32Array} probs Speaker probabilities of the frames, of shape `[num_frames, num_speakers]`.
     * @param {Tensor} silence_embeds Learned silence embedding of shape `[hidden_size]`.
     * @returns {[Float32Array, Float32Array]} The kept frames and their speaker probabilities.
     * @private
     */
    _compress(embeds, probs, silence_embeds) {
        const { num_speakers, speaker_cache_length } = this;
        const num_frames = probs.length / num_speakers;
        const hidden_size = embeds.length / num_frames;

        const scores = this._get_frame_scores(probs);
        // frames beyond the cache capacity are the ones popped from fifo
        for (let i = speaker_cache_length * num_speakers; i < scores.length; ++i) {
            scores[i] += this.latest_frames_score_boost;
        }
        this._boost_scores(scores, this.num_strong_boosted_frames, -2 * LOG_HALF);
        this._boost_scores(scores, this.num_weak_boosted_frames, -LOG_HALF);

        // Speaker-major, each speaker's frames followed by its silence frames, which are always kept
        const num_scored_frames = num_frames + this.num_silence_frames;
        const flat_scores = new Float64Array(num_speakers * num_scored_frames).fill(Infinity);
        for (let t = 0; t < num_frames; ++t) {
            for (let s = 0; s < num_speakers; ++s) {
                flat_scores[s * num_scored_frames + t] = scores[t * num_speakers + s];
            }
        }

        // The best (speaker, frame) pairs, in speaker then frame order. The slots left when fewer than
        // `speaker_cache_length` pairs score above `-Infinity` hold the silence embedding too.
        const kept = topk_indices(flat_scores, speaker_cache_length);
        const silence_data = /** @type {Float32Array} */ (silence_embeds.data);
        const kept_embeds = new Float32Array(speaker_cache_length * hidden_size);
        const kept_probs = new Float32Array(speaker_cache_length * num_speakers);
        for (let i = 0; i < speaker_cache_length; ++i) {
            const t = i < kept.length ? kept[i] % num_scored_frames : num_frames;
            if (t < num_frames) {
                kept_embeds.set(embeds.subarray(t * hidden_size, (t + 1) * hidden_size), i * hidden_size);
                kept_probs.set(probs.subarray(t * num_speakers, (t + 1) * num_speakers), i * num_speakers);
            } else {
                kept_embeds.set(silence_data, i * hidden_size);
            }
        }
        return [kept_embeds, kept_probs];
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
 * import { AutoProcessor, AutoModelForAudioFrameClassification, load_audio, cat } from '@huggingface/transformers';
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
 * // [{ Start: 0.35, End: 9.31, Speaker: 0 }, { Start: 9.15, End: 13.81, Speaker: 1 }, ...]
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
 * // [{ Start: 0.36, End: 9.31, Speaker: 0 }, { Start: 9.17, End: 13.83, Speaker: 1 }, ...]
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
     * `config.fifo_length`; no cache is returned. A session of a single chunk (a recording shorter than the first chunk)
     * passes neither, so it runs offline, which scores it the same way: as one chunk, with an empty cache.
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

        const subsampling_factor = config.audio_config.subsampling_factor;
        const hidden_size = config.audio_config.hidden_size;
        const [batch_size, num_frames] = input_features.dims;
        const num_embeds = Math.ceil(num_frames / subsampling_factor);
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
                    embed_mask[b * num_embeds + t] = mask_data[b * num_frames + t * subsampling_factor] ? 1n : 0n;
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
            const chunk_features = input_features.slice(
                null,
                [start_idx * subsampling_factor, Math.min(stop_idx * subsampling_factor, num_frames)],
                null,
            );

            const cached_embeds = speaker_cache.get_embeds(batch_size, hidden_size);
            const cached_length = cached_embeds.dims[1];

            // the cached frames are always valid
            const step_length = cached_length + stop_idx - start_idx;
            const step_mask = new BigInt64Array(batch_size * step_length).fill(1n);
            for (let b = 0; b < batch_size; ++b) {
                step_mask.set(
                    embed_mask.subarray(b * num_embeds + start_idx, b * num_embeds + stop_idx),
                    b * step_length + cached_length,
                );
            }

            const outputs = await sessionRun(session, {
                input_features: chunk_features,
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

            const start_logit_idx = cached_length * subsampling_factor;
            const end_logit_idx = (cached_length + num_chunk_frames) * subsampling_factor;
            logits.push(outputs.logits.slice(null, [start_logit_idx, end_logit_idx], null));
        }

        return new Nemotron3DiarizationOutput({
            // with no look-ahead, the last encoder frame may be the padding added by feature stacking
            logits: cat(logits, 1).slice(null, [0, num_frames], null),
            speaker_cache: is_streaming ? speaker_cache : null,
        });
    }
}
