import { AutoFeatureExtractor } from '../auto/feature_extraction_auto.js';
import { AutoTokenizer } from '../auto/tokenization_auto.js';
import { Processor } from '../../processing_utils.js';
import { Tensor } from '../../utils/tensor.js';
import { validate_audio_inputs } from '../../feature_extraction_utils.js';

// Voxtral Realtime audio config constants (from mistral_common AudioConfig)
const NUM_LEFT_PAD_TOKENS = 32;
const NUM_DELAY_TOKENS = 6;
const AUDIO_LENGTH_PER_TOK = 8;
const OFFLINE_STREAMING_BUFFER_TOKENS = 10;

/** Token ID for [STREAMING_PAD] in the Voxtral tokenizer. */
const STREAMING_PAD_TOKEN_ID = 32;

export class VoxtralRealtimeProcessor extends Processor {
    static tokenizer_class = AutoTokenizer;
    static feature_extractor_class = AutoFeatureExtractor;
    static uses_processor_config = false;

    /** Number of mel frames in the first audio chunk. */
    get num_mel_frames_first_audio_chunk() {
        return (NUM_DELAY_TOKENS + 1) * AUDIO_LENGTH_PER_TOK;
    }

    /** Number of raw audio samples in the first audio chunk. */
    get num_samples_first_audio_chunk() {
        const { hop_length, n_fft } = this.feature_extractor.config;
        return (this.num_mel_frames_first_audio_chunk - 1) * hop_length + Math.floor(n_fft / 2);
    }

    /** Number of raw audio samples per subsequent audio chunk. */
    get num_samples_per_audio_chunk() {
        const { hop_length, n_fft } = this.feature_extractor.config;
        return AUDIO_LENGTH_PER_TOK * hop_length + n_fft;
    }

    /** Number of right-pad tokens for non-streaming mode. */
    get num_right_pad_tokens() {
        return NUM_DELAY_TOKENS + 1 + OFFLINE_STREAMING_BUFFER_TOKENS;
    }

    /** Number of mel frames per text token. */
    get audio_length_per_tok() {
        return AUDIO_LENGTH_PER_TOK;
    }

    /** Number of raw audio samples per token. */
    get raw_audio_length_per_tok() {
        return AUDIO_LENGTH_PER_TOK * this.feature_extractor.config.hop_length;
    }

    /**
     * Process audio input for VoxtralRealtime.
     *
     * In streaming mode with `is_first_audio_chunk=true`, the audio is left-padded
     * with silence and mel features are extracted with `center=true`.
     * Returns `{ input_ids, input_features }`.
     *
     * In streaming mode with `is_first_audio_chunk=false`, the audio chunk is
     * processed with `center=false` and only `{ input_features }` is returned.
     *
     * In non-streaming mode, the audio is right-padded to ensure the model
     * transcribes the full audio, then processed with `center=true`.
     * Returns `{ input_features }`.
     *
     * @param {Float32Array|Float64Array} audio The audio waveform.
     * @param {Object} [options]
     * @param {boolean} [options.is_streaming=false] Whether processing in streaming mode.
     * @param {boolean} [options.is_first_audio_chunk=true] Whether this is the first audio chunk.
     * @returns {Promise<Object>}
     */
    async _call(audio, { is_streaming = false, is_first_audio_chunk = true } = {}) {
        validate_audio_inputs(audio, 'VoxtralRealtimeProcessor');

        if (!is_streaming && !is_first_audio_chunk) {
            throw new Error('In non-streaming mode (`is_streaming=false`), `is_first_audio_chunk` must be `true`.');
        }

        if (is_first_audio_chunk) {
            if (is_streaming) {
                // Streaming first chunk: left-pad audio with silence, extract mel with center=true, build input_ids
                const num_left_pad_samples = NUM_LEFT_PAD_TOKENS * this.raw_audio_length_per_tok;
                const padded_audio = new Float32Array(num_left_pad_samples + audio.length);
                padded_audio.set(audio, num_left_pad_samples);

                const audio_encoding = await this.feature_extractor(padded_audio, { center: true });

                // Build input_ids: BOS + (num_left_pad_tokens + num_delay_tokens) * [STREAMING_PAD]
                const num_pad_tokens = NUM_LEFT_PAD_TOKENS + NUM_DELAY_TOKENS;
                const num_input_tokens = 1 + num_pad_tokens;
                const input_ids_data = new BigInt64Array(num_input_tokens).fill(BigInt(STREAMING_PAD_TOKEN_ID));
                input_ids_data[0] = 1n; // BOS
                const input_ids = new Tensor('int64', input_ids_data, [1, num_input_tokens]);

                return {
                    input_ids,
                    ...audio_encoding,
                };
            } else {
                // Non-streaming: right-pad audio to ensure full transcription, extract mel with center=true
                const right_pad_samples = this.num_right_pad_tokens * this.raw_audio_length_per_tok;
                const padded_audio = new Float32Array(audio.length + right_pad_samples);
                padded_audio.set(audio);

                return await this.feature_extractor(padded_audio, { center: true });
            }
        } else {
            // Subsequent streaming chunks: extract mel with center=false
            return await this.feature_extractor(audio, { center: false });
        }
    }
}
