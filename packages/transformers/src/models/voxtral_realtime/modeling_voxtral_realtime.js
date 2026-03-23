import { PreTrainedModel } from '../modeling_utils.js';
import { sessionRun } from '../session.js';
import { getCacheShapes } from '../../configs.js';
import { Tensor, ones } from '../../utils/tensor.js';
import { DataTypeMap } from '../../utils/dtypes.js';
import { pick } from '../../utils/core.js';
import { DynamicCache } from '../../cache_utils.js';
import { StoppingCriteria, StoppingCriteriaList } from '../../generation/stopping_criteria.js';

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
    const enc_dtype = encoder_session?.config?.kv_cache_dtype ?? 'float32';
    const enc_cls = enc_dtype === 'float16' ? DataTypeMap.float16 : DataTypeMap.float32;
    const enc_shapes = getCacheShapes(audio_config, { batch_size: 1 });
    for (const name in enc_shapes) {
        const size = enc_shapes[name].reduce((a, b) => a * b, 1);
        enc_kv_cache[name] = new Tensor(enc_dtype, new enc_cls(size), enc_shapes[name]);
    }

    const enc_padding_cache = new Tensor(enc_dtype, new enc_cls(PADDING_CACHE_CHANNELS * CONV1_LEFT_PAD), [
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
        chunks_iter,
        text_hidden_size: text_config.hidden_size,
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
            // Fill audio buffer and embed tokens with audio
            await fillAudioBuffer(enc, enc.audio_consumed + current_len);
        }

        const { inputs_embeds } = await sessionRun(this.sessions['embed_tokens'], { input_ids });
        if (enc) {
            addAudioEmbeddings(enc, inputs_embeds, current_len);
        }

        const decoder_feeds = { inputs_embeds, ...kwargs };
        this.addPastKeyValues(decoder_feeds, past_key_values);

        const session = this.sessions['decoder_model_merged'];
        const fixed = pick(decoder_feeds, session.inputNames);
        return await sessionRun(session, fixed);
    }

    async generate({ input_features, stopping_criteria: userStoppingCriteria, ...kwargs }) {
        if (!input_features) {
            throw new Error('input_features (generator/iterable) must be provided');
        }

        const enc_state = createEncoderState(this, input_features);
        states.set(this, enc_state);

        const stopping_criteria = new StoppingCriteriaList();
        stopping_criteria.push(new AudioExhaustedCriteria(enc_state));
        if (userStoppingCriteria) stopping_criteria.extend(userStoppingCriteria);

        try {
            return await super.generate({ ...kwargs, stopping_criteria });
        } finally {
            // Cleanup encoder state
            enc_state.enc_kv_cache.dispose();
            states.delete(this);
        }
    }
}
