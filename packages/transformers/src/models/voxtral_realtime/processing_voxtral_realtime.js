import { AutoFeatureExtractor } from '../auto/feature_extraction_auto.js';
import { AutoTokenizer } from '../auto/tokenization_auto.js';
import { Processor } from '../../processing_utils.js';
import { Tensor } from '../../utils/tensor.js';

// Streaming constants (from mistral_common AudioConfig + AudioEncoder)
const TOKEN_BOS = 1;
const TOKEN_STREAMING_PAD = 32;
const N_LEFT_PAD_TOKENS = 32;
const N_DELAY_TOKENS = 6;
const RAW_AUDIO_LENGTH_PER_TOK = 1280; // sample_rate (16000) / frame_rate (12.5)
const N_RIGHT_PAD_TOKENS = (N_DELAY_TOKENS + 1) + 10; // 17

export class VoxtralRealtimeProcessor extends Processor {
    static tokenizer_class = AutoTokenizer;
    static feature_extractor_class = AutoFeatureExtractor;
    static uses_processor_config = false;

    /**
     * Process audio for VoxtralRealtime offline transcription.
     *
     * Builds the streaming-format prefix tokens and applies audio padding
     * matching the mistral_common offline streaming format.
     *
     * @param {string|null} _text Unused (kept for API compatibility).
     * @param {Float32Array|Float32Array[]} audio Raw audio samples at 16kHz.
     * @returns {Promise<{input_ids: Tensor, attention_mask: Tensor, audio_values: Tensor}>}
     */
    async _call(_text, audio = null, kwargs = {}) {
        if (!audio) {
            throw new Error('VoxtralRealtimeProcessor requires audio input.');
        }
        if (Array.isArray(audio)) {
            if (audio.length > 1) {
                throw new Error('Batched audio inputs are not supported yet.');
            }
            audio = audio[0];
        }

        // Apply streaming padding (offline mode: left pad + right pad + alignment)
        const paddedAudio = this._pad_audio_streaming(audio);

        // Extract mel spectrogram features
        const features = await this.feature_extractor(paddedAudio);
        const audio_values = features.input_features;

        // Build prefix tokens: [BOS] + [STREAMING_PAD] * (N_LEFT_PAD_TOKENS + N_DELAY_TOKENS)
        const prefixLength = 1 + N_LEFT_PAD_TOKENS + N_DELAY_TOKENS; // 39
        const input_ids_data = new BigInt64Array(prefixLength);
        const attention_mask_data = new BigInt64Array(prefixLength);
        input_ids_data[0] = BigInt(TOKEN_BOS);
        attention_mask_data[0] = 1n;
        for (let i = 1; i < prefixLength; i++) {
            input_ids_data[i] = BigInt(TOKEN_STREAMING_PAD);
            attention_mask_data[i] = 1n;
        }

        return {
            input_ids: new Tensor('int64', input_ids_data, [1, prefixLength]),
            attention_mask: new Tensor('int64', attention_mask_data, [1, prefixLength]),
            audio_values,
        };
    }

    /**
     * Apply streaming padding for offline mode (matching mistral_common AudioEncoder.pad).
     *
     * - Left pad: N_LEFT_PAD_TOKENS * RAW_AUDIO_LENGTH_PER_TOK (32 * 1280 = 40960 samples)
     * - Right pad: alignment to RAW_AUDIO_LENGTH_PER_TOK + N_RIGHT_PAD_TOKENS * RAW_AUDIO_LENGTH_PER_TOK
     *
     * @param {Float32Array} audio Raw audio samples.
     * @returns {Float32Array} Padded audio (silence = zeros).
     */
    _pad_audio_streaming(audio) {
        const leftPad = N_LEFT_PAD_TOKENS * RAW_AUDIO_LENGTH_PER_TOK;
        const alignPad = (RAW_AUDIO_LENGTH_PER_TOK - (audio.length % RAW_AUDIO_LENGTH_PER_TOK)) % RAW_AUDIO_LENGTH_PER_TOK;
        const rightPad = alignPad + N_RIGHT_PAD_TOKENS * RAW_AUDIO_LENGTH_PER_TOK;

        const padded = new Float32Array(leftPad + audio.length + rightPad);
        padded.set(audio, leftPad);
        return padded;
    }
}
