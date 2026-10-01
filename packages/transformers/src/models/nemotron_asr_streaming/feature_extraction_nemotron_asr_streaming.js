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
     * @param {boolean} [options.is_last_audio_chunk=false] Whether the audio is the last chunk of a streaming session.
     * With `center=false`, the end of the audio is then zero-padded by `n_fft / 2 - hop_length` samples, how far the
     * last frame of a centered pass over the whole audio reaches past it at most, so that this frame is kept rather
     * than dropped.
     * @returns {Promise<{ input_features: Tensor; attention_mask: Tensor; }>} The log-Mel features of shape
     * `[1, num_frames, feature_size]`, and the mask of their valid frames.
     */
    async _call(audio, { center = true, is_last_audio_chunk = false } = {}) {
        validate_audio_inputs(audio, 'NemotronAsrStreamingFeatureExtractor');

        const { n_fft, hop_length } = this.config;
        const num_end_padding = !center && is_last_audio_chunk ? Math.floor(n_fft / 2) - hop_length : 0;
        const num_samples = audio.length + num_end_padding;
        if (!center && num_samples < n_fft) {
            throw new Error(
                `Uncentered feature extraction needs at least ${n_fft - num_end_padding} audio samples, got ${audio.length}.`,
            );
        }

        const features = await this._extract_fbank_features(audio, { center, num_end_padding });

        const features_length = center
            ? // Centering pads `n_fft / 2` on each side, so the number of valid frames is `floor(L / hop_length)`.
              Math.floor((num_samples + Math.floor(n_fft / 2) * 2 - n_fft) / hop_length)
            : // No centering: `floor((L - n_fft) / hop_length) + 1` frames, `L` counting the end padding.
              Math.floor((num_samples - n_fft) / hop_length) + 1;

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
