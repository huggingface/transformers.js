import { PreTrainedModel } from '../modeling_utils.js';
import { sessionRun } from '../session.js';
import { getCacheShapes } from '../../configs.js';
import { Tensor, ones } from '../../utils/tensor.js';
import { DataTypeMap } from '../../utils/dtypes.js';
import { LogitsSampler } from '../../generation/logits_sampler.js';
import { DynamicCache } from '../../cache_utils.js';

export class VoxtralRealtimePreTrainedModel extends PreTrainedModel {
    forward_params = ['input_ids', 'attention_mask', 'position_ids', 'input_features', 'past_key_values'];
}

export class VoxtralRealtimeForConditionalGeneration extends VoxtralRealtimePreTrainedModel {
    /**
     * Streaming generation that interleaves audio encoding and text decoding.
     *
     * @param {Object} options
     * @param {Tensor} options.input_ids Initial input token IDs [1, seq_len].
     * @param {Iterable<Tensor>|AsyncIterable<Tensor>} options.input_features Generator/iterable yielding mel spectrogram chunks.
     * @param {number} [options.max_new_tokens=512] Maximum number of new tokens to generate.
     * @param {import('../../generation/configuration_utils.js').GenerationConfig} [options.generation_config=null]
     * @param {import('../../generation/logits_process.js').LogitsProcessorList} [options.logits_processor=null]
     * @param {import('../../generation/stopping_criteria.js').StoppingCriteriaList} [options.stopping_criteria=null]
     * @param {import('../../generation/streamers.js').BaseStreamer} [options.streamer=null]
     * @returns {Promise<Tensor>} Generated token IDs.
     */
    async generate({
        inputs = null,
        input_features = null,
        generation_config = null,
        logits_processor = null,
        stopping_criteria = null,
        streamer = null,
        ...kwargs
    }) {
        this._validate_model_class();

        generation_config = this._prepare_generation_config(generation_config, kwargs);

        let input_ids = kwargs.input_ids ?? inputs;
        if (!(input_ids instanceof Tensor)) {
            throw new Error('input_ids must be provided as a Tensor');
        }
        if (!input_features) {
            throw new Error('input_features (generator/iterable) must be provided');
        }

        // Extract config values
        const { text_config, audio_config } = /** @type {any} */ (this.config);

        const text_hidden_size = text_config.hidden_size;

        const num_mel_bins = audio_config.num_mel_bins;
        const enc_hidden_size = audio_config.hidden_size;

        // Causal conv padding constants
        const CONV1_LEFT_PAD = 2;
        const CONV2_LEFT_PAD = 1;
        const PADDING_CACHE_CHANNELS = num_mel_bins + enc_hidden_size;

        // Sessions
        const encoder_session = this.sessions['audio_encoder'];
        const decoder_session = this.sessions['decoder_model_merged'];
        const embed_session = this.sessions['embed_tokens'];

        // Initialize encoder KV cache using getCacheShapes for consistency with GPU pinning
        const enc_kv_cache = new DynamicCache();
        let enc_padding_cache;
        {
            const enc_dtype = encoder_session?.config?.kv_cache_dtype ?? 'float32';
            const enc_cls = enc_dtype === 'float16' ? DataTypeMap.float16 : DataTypeMap.float32;
            const enc_shapes = getCacheShapes(audio_config, { batch_size: 1 });
            for (const name in enc_shapes) {
                const size = enc_shapes[name].reduce((a, b) => a * b, 1);
                enc_kv_cache[name] = new Tensor(enc_dtype, new enc_cls(size), enc_shapes[name]);
            }

            enc_padding_cache = new Tensor(enc_dtype, new enc_cls(PADDING_CACHE_CHANNELS * CONV1_LEFT_PAD), [
                1,
                PADDING_CACHE_CHANNELS,
                CONV1_LEFT_PAD,
            ]);
        }
        let enc_past_seq_len = 0;

        // Initialize decoder KV cache using getCacheShapes for consistency with GPU pinning
        const decoder_kv = new DynamicCache();
        {
            const dec_dtype = decoder_session?.config?.kv_cache_dtype ?? 'float32';
            const dec_cls = dec_dtype === 'float16' ? DataTypeMap.float16 : DataTypeMap.float32;
            const dec_shapes = getCacheShapes(this.config, { batch_size: 1 });
            for (const name in dec_shapes) {
                const size = dec_shapes[name].reduce((a, b) => a * b, 1);
                decoder_kv[name] = new Tensor(dec_dtype, new dec_cls(size), dec_shapes[name]);
            }
        }

        // Audio embedding queue — chunks are consumed front-to-back, then discarded
        /** @type {{data: import('../../utils/tensor.js').DataArray, tokens: number}[]} */
        const audio_embed_queue = [];
        let audio_embed_total_tokens = 0;
        /** Offset into the first queue entry (partially consumed) */
        let audio_queue_offset = 0;
        /** Total audio embed tokens consumed so far */
        let audio_consumed = 0;
        let stream_exhausted = false;

        // Get iterator from the input_features generator/iterable
        const chunks_iter = input_features[Symbol.asyncIterator]?.() ?? input_features[Symbol.iterator]?.();
        if (!chunks_iter) {
            throw new Error('input_features must be iterable or async iterable');
        }

        // Helper: encode one audio chunk
        async function encodeChunk(chunk_features) {
            const audio_seq_len = chunk_features.dims[2];
            const conv2_output_len = Math.floor((CONV2_LEFT_PAD + audio_seq_len - 3) / 2) + 1;

            const position_ids = new Tensor(
                'int64',
                BigInt64Array.from({ length: conv2_output_len }, (_, i) => BigInt(enc_past_seq_len + i)),
                [1, conv2_output_len],
            );

            const total_seq_len = enc_past_seq_len + conv2_output_len;
            const attention_mask = ones([1, total_seq_len]);

            const { audio_embeds, present_padding_cache, ...present_cache } = await sessionRun(encoder_session, {
                input_features: chunk_features,
                attention_mask,
                position_ids,
                past_padding_cache: enc_padding_cache,
                ...enc_kv_cache,
            });
            enc_padding_cache = present_padding_cache;

            // Update encoder KV cache
            for (const name in present_cache) {
                if (name.startsWith('present.')) {
                    enc_kv_cache[name.replace('present', 'past_key_values')] = present_cache[name];
                }
            }
            enc_past_seq_len = total_seq_len;

            return audio_embeds;
        }

        // Helper: fill audio buffer until it has `needed` tokens
        async function fillAudioBuffer(needed) {
            while (audio_embed_total_tokens < needed && !stream_exhausted) {
                const result = await chunks_iter.next();
                if (result.done) {
                    stream_exhausted = true;
                    break;
                }
                const new_embeds = await encodeChunk(result.value);
                audio_embed_queue.push({ data: new_embeds.data, tokens: new_embeds.dims[1] });
                audio_embed_total_tokens += new_embeds.dims[1];
            }
        }

        // Prepare generation
        let dec_seq_len = input_ids.dims[1];
        let current_input_ids = input_ids;

        const input_ids_length = input_ids.dims[1];
        if (generation_config.max_new_tokens !== null) {
            generation_config.max_length = input_ids_length + generation_config.max_new_tokens;
        }

        // @ts-expect-error ts(2341)
        const prepared_logits_processor = this._get_logits_processor(
            generation_config,
            input_ids_length,
            logits_processor,
        );
        const prepared_stopping_criteria = this._get_stopping_criteria(generation_config, stopping_criteria);

        const sampler = LogitsSampler.getSampler(generation_config);

        /** @type {bigint[][]} */
        const all_input_ids = input_ids.tolist();
        if (streamer) {
            streamer.put(all_input_ids);
        }

        const max_new_tokens = generation_config.max_new_tokens ?? 512;

        // Generation loop
        for (let step = 0; step < max_new_tokens; ++step) {
            const current_len = current_input_ids.dims[1];
            const needed = audio_consumed + current_len;

            // 1. Encode audio chunks until we have enough
            await fillAudioBuffer(needed);

            // 2. Embed tokens
            const { inputs_embeds } = await sessionRun(embed_session, {
                input_ids: current_input_ids,
            });

            // 3. Add audio embeddings for current position(s)
            // Consume from the front of the queue, advancing audio_queue_offset
            if (audio_embed_queue.length > 0) {
                const embed_data = inputs_embeds.data;
                let embed_write_pos = 0;
                let remaining = current_len;

                while (remaining > 0 && audio_embed_queue.length > 0) {
                    const front = audio_embed_queue[0];
                    const available = front.tokens - audio_queue_offset;
                    const n = Math.min(remaining, available);

                    const src_offset = audio_queue_offset * text_hidden_size;
                    for (let i = 0; i < n * text_hidden_size; ++i) {
                        embed_data[embed_write_pos * text_hidden_size + i] += front.data[src_offset + i];
                    }

                    embed_write_pos += n;
                    remaining -= n;
                    audio_queue_offset += n;

                    // If we've fully consumed this chunk, discard it
                    if (audio_queue_offset >= front.tokens) {
                        audio_embed_queue.shift();
                        audio_queue_offset = 0;
                    }
                }
                audio_consumed += current_len - remaining;
            }

            // 4. Run decoder
            const decoder_outputs = await sessionRun(decoder_session, {
                inputs_embeds,
                attention_mask: ones([1, dec_seq_len]),
                ...decoder_kv,
            });

            // Extract logits and update decoder KV cache
            const logits = decoder_outputs.logits;
            for (const name in decoder_outputs) {
                if (name.startsWith('present')) {
                    const pastName = name.replace('present', 'past_key_values');
                    if (pastName in decoder_kv) {
                        const old = decoder_kv[pastName];
                        if (old.location === 'gpu-buffer') old.dispose();
                        decoder_kv[pastName] = decoder_outputs[name];
                    }
                }
            }

            // 5. Sample next token
            const last_logits = logits.slice(null, -1, null).to('float32');
            const next_tokens_scores = prepared_logits_processor(all_input_ids, last_logits);

            const sampledTokens = await sampler(next_tokens_scores[0]);
            const newTokenId = sampledTokens[0][0];
            all_input_ids[0].push(newTokenId);

            if (streamer) {
                streamer.put([[newTokenId]]);
            }

            // 6. Check stopping criteria
            if (stream_exhausted && audio_embed_queue.length === 0) {
                break;
            }

            const stop = prepared_stopping_criteria(all_input_ids);
            if (stop.every((x) => x)) {
                break;
            }

            // 7. Update for next step
            current_input_ids = new Tensor('int64', new BigInt64Array([newTokenId]), [1, 1]);
            dec_seq_len += 1;
        }

        if (streamer) {
            streamer.end();
        }

        // Dispose KV caches
        decoder_kv.dispose();
        enc_kv_cache.dispose();
        if (enc_padding_cache.location === 'gpu-buffer') {
            enc_padding_cache.dispose();
        }

        return new Tensor('int64', BigInt64Array.from(all_input_ids.flat()), [
            all_input_ids.length,
            all_input_ids[0].length,
        ]);
    }
}
