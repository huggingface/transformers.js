import { FeatureExtractor, validate_audio_inputs } from '../../feature_extraction_utils.js';
import { mel_filter_bank, spectrogram, window_function } from '../../utils/audio.js';
import { Tensor } from '../../utils/tensor.js';

export class GraniteSpeechFeatureExtractor extends FeatureExtractor {
    constructor(config) {
        super(config);

        const { n_fft, win_length, n_mels, sample_rate } = config.melspec_kwargs;

        // torchaudio uses HTK mel scale with no norm by default
        this.mel_filters = mel_filter_bank(
            Math.floor(1 + n_fft / 2), // num_frequency_bins = 257
            n_mels, // 80
            0, // min_frequency
            sample_rate / 2, // max_frequency = 8000
            sample_rate, // 16000
            null, // norm (torchaudio default: no norm)
            'htk', // mel_scale (torchaudio default)
        );

        // torchaudio center-pads the window when win_length < n_fft:
        // pad_amount = (n_fft - win_length) // 2 on each side
        const raw_window = window_function(win_length, 'hann');
        this.window = new Float64Array(n_fft);
        const pad = Math.floor((n_fft - win_length) / 2);
        this.window.set(raw_window, pad);
    }

    /**
     * Extract mel spectrogram features from audio, matching the Python GraniteSpeechFeatureExtractor.
     * @param {Float32Array|Float64Array} audio The audio waveform.
     * @returns {Promise<{input_features: Tensor}>}
     */
    async _call(audio) {
        validate_audio_inputs(audio, 'GraniteSpeechFeatureExtractor');

        const { n_fft, hop_length, n_mels } = this.config.melspec_kwargs;

        const mel = await spectrogram(audio, this.window, n_fft, hop_length, {
            power: 2.0,
            mel_filters: this.mel_filters,
            log_mel: 'log10',
        });
        // mel shape: [n_mels, num_frames]

        const num_frames = mel.dims[1];
        const data = /** @type {Float32Array} */ (mel.data);

        // Find global max for normalization
        let mx = -Infinity;
        for (let i = 0; i < data.length; ++i) {
            if (data[i] > mx) mx = data[i];
        }

        // Apply: max(logmel, mx - 8.0) / 4 + 1
        const threshold = mx - 8.0;
        for (let i = 0; i < data.length; ++i) {
            data[i] = Math.max(data[i], threshold) / 4 + 1;
        }

        // Transpose [n_mels, time] → [time, n_mels], removing last frame if odd
        const time = num_frames % 2 === 1 ? num_frames - 1 : num_frames;
        const transposed = new Float32Array(time * n_mels);
        for (let t = 0; t < time; ++t) {
            for (let m = 0; m < n_mels; ++m) {
                transposed[t * n_mels + m] = data[m * num_frames + t];
            }
        }

        // Stack adjacent frame pairs: [time, n_mels] → [time/2, 2*n_mels]
        // In Python: logmel.reshape(bsz, -1, 2 * logmel.shape[-1])
        // Adjacent rows [2t, 2t+1] are concatenated along the feature dim
        const stacked_time = time / 2;
        const stacked_dim = 2 * n_mels;
        const stacked = new Float32Array(stacked_time * stacked_dim);
        for (let t = 0; t < stacked_time; ++t) {
            const src_offset = 2 * t * n_mels;
            const dst_offset = t * stacked_dim;
            stacked.set(transposed.subarray(src_offset, src_offset + stacked_dim), dst_offset);
        }

        const input_features = new Tensor('float32', stacked, [1, stacked_time, stacked_dim]);

        return { input_features };
    }
}
