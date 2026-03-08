import { PreTrainedModel } from '../modeling_utils.js';
import { sessionRun } from '../session.js';
import { getCacheShapes } from '../../configs.js';
import { Tensor, cat, ones } from '../../utils/tensor.js';
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

        // Audio embedding buffer
        let audio_embed_buffer = new Tensor('float32', new Float32Array(0), [1, 0, text_hidden_size]);
        let audio_pos = 0;
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
            const attention_mask = new Tensor('int64', new BigInt64Array(total_seq_len).fill(1n), [1, total_seq_len]);

            const encoder_inputs = {
                input_features: chunk_features,
                attention_mask,
                position_ids,
                past_padding_cache: enc_padding_cache,
                ...enc_kv_cache,
            };

            const outputs = await sessionRun(encoder_session, encoder_inputs);

            // Extract outputs by name
            const output_names = encoder_session.outputNames;
            const audio_embeds = outputs[output_names[0]];
            enc_padding_cache = outputs['present_padding_cache'];

            // Update encoder KV cache
            for (const name in outputs) {
                if (name.startsWith('present.')) {
                    enc_kv_cache[name.replace('present', 'past_key_values')] = outputs[name];
                }
            }
            enc_past_seq_len = total_seq_len;

            return audio_embeds;
        }

        // Helper: fill audio buffer until it has `needed` tokens
        async function fillAudioBuffer(needed) {
            while (audio_embed_buffer.dims[1] < needed && !stream_exhausted) {
                const result = await chunks_iter.next();
                if (result.done) {
                    stream_exhausted = true;
                    break;
                }
                const new_embeds = await encodeChunk(result.value);
                if (audio_embed_buffer.dims[1] === 0) {
                    audio_embed_buffer = new_embeds;
                } else {
                    audio_embed_buffer = cat([audio_embed_buffer, new_embeds], 1);
                }
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
            const needed = audio_pos + current_len;

            // 1. Encode audio chunks until we have enough
            await fillAudioBuffer(needed);

            // 2. Embed tokens
            const { inputs_embeds } = await sessionRun(embed_session, {
                input_ids: current_input_ids,
            });

            // 3. Add audio embeddings for current position(s)
            const num_audio_tokens = audio_embed_buffer.dims[1];
            if (audio_pos < num_audio_tokens) {
                const end_pos = Math.min(audio_pos + current_len, num_audio_tokens);
                const n = end_pos - audio_pos;

                // Add audio embeddings to text embeddings in-place
                const embed_data = inputs_embeds.data;
                const audio_data = audio_embed_buffer.data;
                const offset_audio = audio_pos * text_hidden_size;
                for (let i = 0; i < n * text_hidden_size; ++i) {
                    embed_data[i] += audio_data[offset_audio + i];
                }
                audio_pos = end_pos;
            }

            // 4. Run decoder
            const decoder_inputs = {
                inputs_embeds,
                attention_mask: ones([1, dec_seq_len]),
                ...decoder_kv,
            };
            const decoder_outputs = await sessionRun(decoder_session, decoder_inputs);

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

            // 6. Sample next token
            const last_logits = logits.slice(null, -1, null).to('float32');
            const next_tokens_scores = prepared_logits_processor(all_input_ids, last_logits);

            const sampledTokens = await sampler(next_tokens_scores[0]);
            const newTokenId = sampledTokens[0][0];
            all_input_ids[0].push(newTokenId);
            const generated_input_ids = [[newTokenId]];

            if (streamer) {
                streamer.put(generated_input_ids);
            }

            // 7. Check stopping criteria
            if (stream_exhausted && audio_pos >= num_audio_tokens) {
                break;
            }

            const stop = prepared_stopping_criteria(all_input_ids);
            if (stop.every((x) => x)) {
                break;
            }

            // 8. Update for next step
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
        if (audio_embed_buffer.location === 'gpu-buffer') {
            audio_embed_buffer.dispose();
        }

        return new Tensor('int64', BigInt64Array.from(all_input_ids.flat()), [
            all_input_ids.length,
            all_input_ids[0].length,
        ]);
    }
}
