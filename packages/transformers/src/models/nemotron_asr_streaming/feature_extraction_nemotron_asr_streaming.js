import { validate_audio_inputs } from '../../feature_extraction_utils.js';
import { Tensor } from '../../utils/tensor.js';
import { ParakeetFeatureExtractor } from '../parakeet/feature_extraction_parakeet.js';

/**
 * Feature extractor for NVIDIA's streaming NeMo models (e.g., Nemotron ASR Streaming and Nemotron-3-Diarization).
 *
 * Extracts the same log-Mel spectrogram as {@link ParakeetFeatureExtractor}, but never normalizes it, and can
 * disable window centering so that streaming chunks reproduce, frame for frame, a single pass over the whole audio.
 */
export class NemotronAsrStreamingFeatureExtractor extends ParakeetFeatureExtractor {
    /**
     * Extracts the log-Mel spectrogram of the given audio.
     * @param {Float32Array|Float64Array} audio The audio data as a Float32Array/Float64Array.
     * @param {Object} [options]
     * @param {boolean} [options.center=true] Whether to pad the audio on both sides so that the STFT frames are
     * centered. Use `true` for offline extraction and for the first chunk of a streaming session, and `false` for
     * the later chunks: feeding `audio[hop_length * frame - n_fft / 2, ...]` with `center=false` reproduces the
     * frames that a single centered pass over the whole audio would have produced.
     * @returns {Promise<{ input_features: Tensor; attention_mask: Tensor; }>} The log-Mel features of shape
     * `[1, num_frames, feature_size]`, and the mask of their valid frames.
     */
    async _call(audio, { center = true } = {}) {
        validate_audio_inputs(audio, 'NemotronAsrStreamingFeatureExtractor');

        const { n_fft, hop_length } = this.config;
        if (!center && audio.length < n_fft) {
            throw new Error(
                `Uncentered feature extraction needs at least \`n_fft\` (${n_fft}) audio samples, got ${audio.length}.`,
            );
        }

        const features = await this._extract_fbank_features(audio, { center });

        const features_length = center
            ? // Centering pads `n_fft / 2` on each side, so the number of valid frames is `floor(L / hop_length)`.
              Math.floor((audio.length + Math.floor(n_fft / 2) * 2 - n_fft) / hop_length)
            : // No padding: `floor((L - n_fft) / hop_length) + 1` frames.
              Math.floor((audio.length - n_fft) / hop_length) + 1;

        const [num_frames, num_features] = features.dims;
        /** @type {Float32Array} */ (features.data).fill(0, features_length * num_features);

        const mask_data = new BigInt64Array(num_frames);
        mask_data.fill(1n, 0, features_length);

        return {
            input_features: features.unsqueeze_(0),
            attention_mask: new Tensor('int64', mask_data, [1, num_frames]),
        };
    }
}
