import { PreTrainedModel, addPastKeyValues, resolveCacheShape } from '../modeling_utils.js';
import { sessionRun } from '../session.js';
import { getCacheNames } from '../../configs.js';
import { Tensor, ones } from '../../utils/tensor.js';
import { DataTypeMap } from '../../utils/dtypes.js';
import { pick } from '../../utils/core.js';
import { DynamicCache } from '../../cache_utils.js';
import { StoppingCriteria, StoppingCriteriaList } from '../../generation/stopping_criteria.js';
import { LogitsProcessor, LogitsProcessorList } from '../../generation/logits_process.js';

// Causal conv padding constants
const CONV1_LEFT_PAD = 2;
const CONV2_LEFT_PAD = 1;

/**
 * WeakMap to hold encoder streaming states for each model instance during generation.
 * This allows the state to be accessed and modified across the generation process
 * without exposing it on the model instance itself.
 * @private
 * @type {WeakMap<VoxtralRealtimeForConditionalGeneration, Object>}
 */
const states = new WeakMap();

/**
 * Creates encoder streaming state for a VoxtralRealtime generation session.
 * @param {VoxtralRealtimeForConditionalGeneration} model
 * @param {Iterable<Tensor>|AsyncIterable<Tensor>} input_features
 * @returns {Object} Encoder state object.
 * @private
 */
function createEncoderState(model, input_features) {
    const { text_config, audio_config } = /** @type {any} */ (model.config);
    const encoder_session = model.sessions['audio_encoder'];

    const { num_mel_bins, hidden_size: enc_hidden_size } = audio_config;
    const PADDING_CACHE_CHANNELS = num_mel_bins + enc_hidden_size;

    // Initialize encoder KV cache
    const enc_kv_cache = new DynamicCache();
    const enc_names = getCacheNames(audio_config);
    const enc_symbols = { batch_size: 1 };
    /** @type {import('onnxruntime-common').Tensor.Type} */
    let padding_type = 'float32';
    for (const meta of encoder_session.inputMetadata) {
        if (meta.name === 'past_padding_cache') {
            padding_type = meta.type;
            continue;
        }
        if (!enc_names.has(meta.name)) continue;
        const shape = resolveCacheShape(meta.shape, enc_symbols);
        const size = shape.reduce((a, b) => a * b, 1);
        const cls = DataTypeMap[meta.type];
        enc_kv_cache[meta.name] = new Tensor(meta.type, new cls(size), shape);
    }

    const padding_cls = DataTypeMap[padding_type];
    const enc_padding_cache = new Tensor(padding_type, new padding_cls(PADDING_CACHE_CHANNELS * CONV1_LEFT_PAD), [
        1,
        PADDING_CACHE_CHANNELS,
        CONV1_LEFT_PAD,
    ]);

    // Set up iterator from input_features
    const chunks_iter = input_features[Symbol.asyncIterator]?.() ?? input_features[Symbol.iterator]?.();
    if (!chunks_iter) {
        throw new Error('input_features must be iterable or async iterable');
    }

    return {
        encoder_session,
        enc_kv_cache,
        enc_padding_cache,
        enc_past_seq_len: 0,
        audio_embed_queue: [],
        audio_embed_total_tokens: 0,
        audio_queue_offset: 0,
        audio_consumed: 0,
        stream_exhausted: false,
        /** The step now being decoded had no audio left (`forward` set it): force `</s>`. */
        forcing_eos: false,
        /**
         * The end-of-sequence token(s) of this call, which is what a step without
         * audio has to end on. It is the effective generation config's value, not the
         * model's stored one, so `generate({ eos_token_id })` and a per-call
         * `generation_config` are honoured here too. `generate` fills it in.
         * @type {number|number[]|null}
         */
        eos_token_id: null,
        chunks_iter,
        text_hidden_size: text_config.hidden_size,
        vocab_size: text_config.vocab_size,
    };
}

/**
 * Encodes one audio chunk through the audio encoder.
 * @param {Object} s Encoder state.
 * @param {Tensor} chunk_features Mel spectrogram chunk [1, num_mel_bins, seq_len].
 * @returns {Promise<Tensor>} Audio embeddings.
 * @private
 */
async function encodeChunk(s, chunk_features) {
    const audio_seq_len = chunk_features.dims[2];
    const conv2_output_len = Math.floor((CONV2_LEFT_PAD + audio_seq_len - 3) / 2) + 1;

    const position_ids = new Tensor(
        'int64',
        BigInt64Array.from({ length: conv2_output_len }, (_, i) => BigInt(s.enc_past_seq_len + i)),
        [1, conv2_output_len],
    );

    const total_seq_len = s.enc_past_seq_len + conv2_output_len;
    const attention_mask = ones([1, total_seq_len]);
    const { audio_embeds, present_padding_cache, ...present_cache } = await sessionRun(s.encoder_session, {
        input_features: chunk_features,
        attention_mask,
        position_ids,
        past_padding_cache: s.enc_padding_cache,
        ...s.enc_kv_cache,
    });
    // Dispose previous padding cache and update
    if (s.enc_padding_cache.location === 'gpu-buffer') {
        s.enc_padding_cache.dispose();
    }
    s.enc_padding_cache = present_padding_cache;

    // Update encoder KV cache, disposing previous tensors
    for (const name in present_cache) {
        if (name.startsWith('present.')) {
            const pastName = name.replace('present', 'past_key_values');
            const prev = s.enc_kv_cache[pastName];
            if (prev?.location === 'gpu-buffer') {
                prev.dispose();
            }
            s.enc_kv_cache[pastName] = present_cache[name];
        }
    }
    s.enc_past_seq_len = total_seq_len;
    return audio_embeds;
}

/**
 * Fills the audio embedding buffer until it has enough tokens.
 * @param {Object} s Encoder state.
 * @param {number} needed Total number of audio tokens needed.
 * @private
 */
async function fillAudioBuffer(s, needed) {
    while (s.audio_embed_total_tokens < needed && !s.stream_exhausted) {
        const result = await s.chunks_iter.next();
        if (result.done) {
            s.stream_exhausted = true;
            break;
        }
        const new_embeds = await encodeChunk(s, result.value);
        s.audio_embed_queue.push({ data: new_embeds.data, tokens: new_embeds.dims[1] });
        s.audio_embed_total_tokens += new_embeds.dims[1];
    }
}

/**
 * Adds audio embeddings to text embeddings from the queue.
 * @param {Object} s Encoder state.
 * @param {Tensor} inputs_embeds Text embeddings tensor (modified in-place).
 * @param {number} current_len Number of tokens to consume.
 * @private
 */
function addAudioEmbeddings(s, inputs_embeds, current_len) {
    if (s.audio_embed_queue.length === 0) return;

    const embed_data = inputs_embeds.data;
    let embed_write_pos = 0;
    let remaining = current_len;

    while (remaining > 0 && s.audio_embed_queue.length > 0) {
        const front = s.audio_embed_queue[0];
        const available = front.tokens - s.audio_queue_offset;
        const n = Math.min(remaining, available);

        const src_offset = s.audio_queue_offset * s.text_hidden_size;
        for (let i = 0; i < n * s.text_hidden_size; ++i) {
            embed_data[embed_write_pos * s.text_hidden_size + i] += front.data[src_offset + i];
        }

        embed_write_pos += n;
        remaining -= n;
        s.audio_queue_offset += n;

        if (s.audio_queue_offset >= front.tokens) {
            s.audio_embed_queue.shift();
            s.audio_queue_offset = 0;
        }
    }
    s.audio_consumed += current_len - remaining;
}

/**
 * Logits that put all probability on the end-of-sequence token, for a decoding step
 * whose audio has run out.
 *
 * The decoder's input at every position is the sum of an audio embedding and a text
 * embedding; it was never trained on a text embedding alone, and what it samples from
 * one is arbitrary (a word-piece suffix such as "ished", or punctuation). The Python
 * implementation never runs such a step: it stops as soon as the chunk generator is
 * exhausted. Here the loop can only learn that the stream has ended once it needs the
 * next chunk, so the step is answered with `</s>` instead of being decoded.
 *
 * A call without an end-of-sequence token can only fail at that step, never earlier:
 * until the audio runs out, nothing here is needed.
 * @param {Object} s Encoder state.
 * @param {number} seq_len Number of positions in the current step.
 * @returns {Tensor} Logits of shape `[1, seq_len, vocab_size]`.
 * @private
 */
function endOfSequenceLogits(s, seq_len) {
    const { vocab_size, eos_token_id } = s;
    if (eos_token_id == null) {
        throw new Error('VoxtralRealtime needs an `eos_token_id` to end generation when the audio runs out.');
    }
    const eos = Array.isArray(eos_token_id) ? eos_token_id[0] : eos_token_id;
    const data = new Float32Array(seq_len * vocab_size).fill(-Infinity);
    for (let i = 0; i < seq_len; ++i) {
        data[i * vocab_size + eos] = 0;
    }
    return new Tensor('float32', data, [1, seq_len, vocab_size]);
}

/**
 * Forces the end-of-sequence token on a step whose audio has run out.
 *
 * `forward` already answers such a step with `</s>` logits, but those are only an
 * input to the processors that run after it: `min_new_tokens` (and `min_length`)
 * mask every end-of-sequence token until enough have been generated, which would
 * leave a row of all `-Infinity` — `NaN` after the softmax, or token 0 from an
 * argmax — and generation would carry on decoding audio-less steps. This processor
 * is appended last, so it runs after all of them.
 *
 * Its `eos_token_id` is the one on the encoder state, which is the effective generation
 * config of this call, not the model's stored one: `generate({ eos_token_id })` and a
 * per-call `generation_config` replace it, and a stop token the helper above never read
 * would leave generation running to `max_new_tokens`.
 * @private
 */
class AudioExhaustedLogitsProcessor extends LogitsProcessor {
    /**
     * @param {Object} enc_state The encoder streaming state, which carries the
     * end-of-sequence token(s) of this call.
     */
    constructor(enc_state) {
        super();
        this._s = enc_state;
    }

    _call(input_ids, logits) {
        if (!this._s.forcing_eos) return logits;
        const { eos_token_id } = this._s;
        const eos_tokens = Array.isArray(eos_token_id) ? eos_token_id : [eos_token_id];
        for (let i = 0; i < logits.dims[0]; ++i) {
            const row = /** @type {Float32Array} */ (logits[i].data);
            row.fill(-Infinity);
            for (const eos_token of eos_tokens) {
                row[eos_token] = 0;
            }
        }
        return logits;
    }
}

/**
 * Stopping criterion that triggers when the audio stream is exhausted
 * and all buffered audio embeddings have been consumed.
 * @private
 */
class AudioExhaustedCriteria extends StoppingCriteria {
    constructor(enc_state) {
        super();
        this._s = enc_state;
    }
    _call(input_ids) {
        const done = this._s.stream_exhausted && this._s.audio_embed_queue.length === 0;
        return input_ids.map(() => done);
    }
}

export class VoxtralRealtimePreTrainedModel extends PreTrainedModel {
    forward_params = ['input_ids', 'attention_mask', 'position_ids', 'past_key_values'];
}

export class VoxtralRealtimeForConditionalGeneration extends VoxtralRealtimePreTrainedModel {
    async forward({ input_ids, past_key_values, ...kwargs }) {
        const current_len = input_ids.dims[1];

        const enc = states.get(this);
        if (enc) {
            enc.forcing_eos = false;
            // Fill audio buffer and embed tokens with audio
            await fillAudioBuffer(enc, enc.audio_consumed + current_len);
            if (enc.audio_embed_total_tokens - enc.audio_consumed < current_len) {
                // The stream ended before this step: end the sequence rather than decode
                // positions that have no audio embedding. The flag makes
                // `AudioExhaustedLogitsProcessor` force `</s>` whatever the other
                // processors did to these logits; the cache is not advanced here, and
                // does not need to be, because generation stops on that token.
                enc.forcing_eos = true;
                return { logits: endOfSequenceLogits(enc, current_len) };
            }
        }

        const { inputs_embeds } = await sessionRun(this.sessions['embed_tokens'], { input_ids });
        if (enc) {
            addAudioEmbeddings(enc, inputs_embeds, current_len);
        }

        const decoder_feeds = { inputs_embeds, ...kwargs };
        addPastKeyValues(this, decoder_feeds, past_key_values);

        const session = this.sessions['decoder_model_merged'];
        const fixed = pick(decoder_feeds, session.inputNames);
        return await sessionRun(session, fixed);
    }

    async generate({
        input_features,
        stopping_criteria: userStoppingCriteria,
        logits_processor: userLogitsProcessor,
        generation_config,
        ...kwargs
    }) {
        if (!input_features) {
            throw new Error('input_features (generator/iterable) must be provided');
        }

        // Resolve the effective generation config before the encoder state is created, so
        // that the state carries the `eos_token_id` this call uses — `generate({ eos_token_id })`
        // and a per-call `generation_config` included. A missing one is not an error yet: a
        // call that stops before the audio runs out never needs it.
        const { eos_token_id } = this._prepare_generation_config(generation_config, kwargs);

        const enc_state = createEncoderState(this, input_features);
        enc_state.eos_token_id = eos_token_id;
        states.set(this, enc_state);

        const stopping_criteria = new StoppingCriteriaList();
        stopping_criteria.push(new AudioExhaustedCriteria(enc_state));
        if (userStoppingCriteria) stopping_criteria.extend(userStoppingCriteria);

        // Ending the step that runs out of audio is a processor of ours, appended last
        // so that it runs after `min_new_tokens` and the rest. The logits `forward`
        // returns for that step are only a fallback for callers that bypass `generate`.
        const logits_processor = new LogitsProcessorList();
        if (userLogitsProcessor) {
            logits_processor.extend(userLogitsProcessor);
        }
        logits_processor.push(new AudioExhaustedLogitsProcessor(enc_state));

        try {
            return await super.generate({ ...kwargs, generation_config, stopping_criteria, logits_processor });
        } finally {
            // Cleanup encoder state
            enc_state.enc_kv_cache.dispose();
            states.delete(this);
        }
    }
}
