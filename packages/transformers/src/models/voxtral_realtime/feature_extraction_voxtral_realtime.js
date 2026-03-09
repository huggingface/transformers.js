import { FeatureExtractor, validate_audio_inputs } from '../../feature_extraction_utils.js';
import { Tensor } from '../../utils/tensor.js';
import { mel_filter_bank, spectrogram, window_function } from '../../utils/audio.js';

export class VoxtralRealtimeFeatureExtractor extends FeatureExtractor {
    constructor(config) {
        super(config);

        const num_frequency_bins = Math.floor(1 + this.config.n_fft / 2);
        this.config.mel_filters ??= mel_filter_bank(
            num_frequency_bins,
            this.config.feature_size, // 128
            0.0,
            8000.0, // max_frequency
            this.config.sampling_rate, // 16000
            'slaney',
            'slaney',
        );

        this.window = window_function(this.config.n_fft, 'hann');
    }

    /**
     * Computes log-Mel spectrogram features for VoxtralRealtime.
     * Uses log10 with global_log_mel_max normalization (same as Whisper but with fixed max).
     * @param {Float32Array|Float64Array} waveform
     * @returns {Promise<Tensor>}
     */
    async _extract_fbank_features(waveform) {
        const features = await spectrogram(
            waveform,
            this.window,
            this.config.n_fft, // 400
            this.config.hop_length, // 160
            {
                power: 2.0,
                mel_filters: this.config.mel_filters,
                log_mel: 'log10',
                max_num_frames: null,
            },
        );

        // Remove the last time frame to match Python's stft[..., :-1]
        let result = features.slice(null, [0, -1]);

        // Apply normalization
        const maxValue = this.config.global_log_mel_max ?? 1.5;
        const data = /** @type {Float32Array} */ (result.data);
        for (let i = 0; i < data.length; ++i) {
            data[i] = (Math.max(data[i], maxValue - 8.0) + 4.0) / 4.0;
        }

        // Pad to a multiple of 8 frames so the encoder's conv2 (stride=2) output
        // is divisible by downsample_factor=4: floor(8k / 2) = 4k
        const mel_frames = result.dims[1];
        const padded_frames = Math.ceil(mel_frames / 8) * 8;
        if (padded_frames !== mel_frames) {
            const [mel_bins] = result.dims;
            const silence = (Math.max(Math.log10(1e-10), maxValue - 8.0) + 4.0) / 4.0;
            const padded_data = new Float32Array(mel_bins * padded_frames).fill(silence);
            // Copy normalized data into the padded buffer
            for (let m = 0; m < mel_bins; m++) {
                padded_data.set(
                    data.subarray(m * mel_frames, (m + 1) * mel_frames),
                    m * padded_frames,
                );
            }
            result = new Tensor('float32', padded_data, [mel_bins, padded_frames]);
        }

        return result;
    }

    /**
     * @param {Float32Array|Float64Array} audio
     * @returns {Promise<{ input_features: Tensor }>}
     */
    async _call(audio) {
        validate_audio_inputs(audio, 'VoxtralRealtimeFeatureExtractor');

        const features = await this._extract_fbank_features(audio);

        return {
            input_features: features.unsqueeze_(0),
        };
    }
}
