import { PreTrainedModel, getPastLength, decoder_forward } from '../modeling_utils.js';
import { sessionRun } from '../session.js';
import { Tensor } from '../../utils/tensor.js';

const ENCODER_NUM_LAYERS = 32;
const ENCODER_NUM_HEADS = 32;
const ENCODER_HEAD_DIM = 64;
const PADDING_CACHE_DIM = 1408; // conv1.in_channels (128) + conv1.out_channels (1280)
const EOS_TOKEN_ID = 2;

export class VoxtralRealtimePreTrainedModel extends PreTrainedModel {
    forward_params = ['input_ids', 'attention_mask', 'position_ids', 'audio_values', 'past_key_values'];
}

export class VoxtralRealtimeForConditionalGeneration extends VoxtralRealtimePreTrainedModel {
    /** @type {Tensor|null} Full audio embeddings [1, n_audio, 3072] */
    _audio_embeds = null;

    /**
     * Encode audio through the ONNX merged encoder (conv + transformer + projector).
     * Returns audio embeddings of shape [batch, n_audio, 3072].
     */
    async encode_audio({ audio_values }) {
        const batch_size = audio_values.dims[0];
        const mel_frames = audio_values.dims[2];
        const seq_len = Math.floor(mel_frames / 2);

        const attention_mask = new Tensor(
            'int64',
            new BigInt64Array(batch_size * seq_len).fill(1n),
            [batch_size, seq_len],
        );
        const position_ids_data = new BigInt64Array(batch_size * seq_len);
        for (let b = 0; b < batch_size; b++) {
            for (let i = 0; i < seq_len; i++) {
                position_ids_data[b * seq_len + i] = BigInt(i);
            }
        }
        const position_ids = new Tensor('int64', position_ids_data, [batch_size, seq_len]);

        const past_padding_cache = new Tensor(
            'float32',
            new Float32Array(batch_size * PADDING_CACHE_DIM * 2),
            [batch_size, PADDING_CACHE_DIM, 2],
        );

        const feeds = { input_features: audio_values, attention_mask, position_ids, past_padding_cache };
        for (let i = 0; i < ENCODER_NUM_LAYERS; i++) {
            const empty = new Tensor(
                'float32',
                new Float32Array(0),
                [batch_size, ENCODER_NUM_HEADS, 0, ENCODER_HEAD_DIM],
            );
            feeds[`past_key_values.${i}.key`] = empty;
            feeds[`past_key_values.${i}.value`] = empty;
        }

        const result = await sessionRun(this.sessions['audio_encoder'], feeds);
        return result.audio_embeds;
    }

    /**
     * Custom forward that adds audio embeddings at EVERY generation step.
     *
     * VoxtralRealtime architecture requires:
     *   embed = audio_embed[pos] + text_embed(token)
     * at every position — not just during prefill like standard multimodal models.
     *
     * Flow:
     *   1. Prefill: audio_embeds[:, :L, :] + embed_tokens(prefix_ids)
     *   2. Each generation step at position pos: audio_embeds[:, pos, :] + embed_tokens(prev_token)
     *   3. Force EOS when pos >= n_audio (all audio consumed)
     */
    async forward(params) {
        const {
            input_ids = null,
            attention_mask = null,
            position_ids = null,
            inputs_embeds = null,
            past_key_values = null,
            generation_config = null,
            logits_processor = null,
            audio_values = null,
            ...kwargs
        } = params;

        let currentEmbeds = inputs_embeds;
        let currentAttentionMask = attention_mask;

        if (!currentEmbeds) {
            currentEmbeds = await this.encode_text({ input_ids, ...kwargs });

            if (audio_values && input_ids.dims[1] !== 1) {
                // ── Prefill ──
                const audioEmbeds = await this.encode_audio({ audio_values });
                this._audio_embeds = audioEmbeds;

                const textData = /** @type {Float32Array} */ (currentEmbeds.data);
                const audioData = /** @type {Float32Array} */ (audioEmbeds.data);
                const hiddenSize = currentEmbeds.dims[2];
                const prefixLen = input_ids.dims[1];
                const count = prefixLen * hiddenSize;

                for (let i = 0; i < count; i++) {
                    textData[i] += audioData[i];
                }
            } else if (past_key_values && this._audio_embeds) {
                // ── Generation step ──
                const pastLength = getPastLength(past_key_values);
                const n_audio = this._audio_embeds.dims[1];

                if (pastLength < n_audio) {
                    const textData = /** @type {Float32Array} */ (currentEmbeds.data);
                    const audioData = /** @type {Float32Array} */ (this._audio_embeds.data);
                    const hiddenSize = currentEmbeds.dims[2];
                    const audioOffset = pastLength * hiddenSize;

                    for (let k = 0; k < hiddenSize; k++) {
                        textData[k] += audioData[audioOffset + k];
                    }
                }
            }
        }

        const outputs = await decoder_forward(
            this,
            {
                inputs_embeds: currentEmbeds,
                past_key_values,
                attention_mask: currentAttentionMask,
                position_ids,
                generation_config,
                logits_processor,
            },
            true,
        );

        // Force EOS when all audio positions are consumed
        if (this._audio_embeds && past_key_values) {
            const pastLength = getPastLength(past_key_values);
            const n_audio = this._audio_embeds.dims[1];
            if (pastLength >= n_audio) {
                const logits = outputs.logits;
                const logitsData = /** @type {Float32Array} */ (logits.data);
                const vocabSize = logits.dims[logits.dims.length - 1];
                const offset = logits.dims.length === 3
                    ? (logits.dims[1] - 1) * vocabSize
                    : 0;
                for (let i = 0; i < vocabSize; i++) {
                    logitsData[offset + i] = (i === EOS_TOKEN_ID) ? 0 : -Infinity;
                }
            }
        }

        return outputs;
    }
}
