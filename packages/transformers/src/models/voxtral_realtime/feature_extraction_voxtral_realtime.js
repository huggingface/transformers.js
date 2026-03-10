import { FeatureExtractor, validate_audio_inputs } from '../../feature_extraction_utils.js';
import { mel_filter_bank, spectrogram, window_function } from '../../utils/audio.js';

export class VoxtralRealtimeFeatureExtractor extends FeatureExtractor {
    constructor(config) {
        super(config);

        this.config.mel_filters ??= mel_filter_bank(
            Math.floor(1 + this.config.n_fft / 2), // num_frequency_bins
            this.config.feature_size, // num_mel_filters
            0.0, // min_frequency
            8000.0, // max_frequency
            this.config.sampling_rate, // sampling_rate
            'slaney', // norm
            'slaney', // mel_scale
        );

        this.window = window_function(this.config.n_fft, 'hann');
    }

    /**
     * Computes the log-Mel spectrogram of the provided audio waveform.
     * @param {Float32Array|Float64Array} waveform The audio waveform to process.
     * @param {Object} [options]
     * @param {boolean} [options.center=true] Whether to center-pad the waveform for STFT.
     * @returns {Promise<import('../../utils/tensor.js').Tensor>} The log-Mel spectrogram tensor of shape [num_mel_bins, num_frames].
     */
    async _extract_fbank_features(waveform, { center = true } = {}) {
        const { n_fft, hop_length, mel_filters, global_log_mel_max } = this.config;

        // torch.stft drops the last frame via [:-1]
        // center=True:  floor(signal_len / hop_length) frames
        // center=False: floor((signal_len - n_fft) / hop_length) frames
        const max_num_frames = center
            ? Math.floor(waveform.length / hop_length)
            : Math.floor((waveform.length - n_fft) / hop_length);

        return await spectrogram(
            waveform,
            this.window,
            n_fft, // frame_length
            hop_length,
            {
                power: 2.0,
                mel_filters,
                log_mel: 'log10_max_norm',
                max_log_mel: global_log_mel_max,
                center,
                max_num_frames,
                do_pad: false,
            },
        );
    }

    /**
     * Extract mel spectrogram features from audio.
     * @param {Float32Array|Float64Array} audio The audio data.
     * @param {Object} [options]
     * @param {boolean} [options.center=true] Whether to center-pad the waveform.
     * @returns {Promise<{ input_features: import('../../utils/tensor.js').Tensor }>}
     */
    async _call(audio, { center = true } = {}) {
        validate_audio_inputs(audio, 'VoxtralRealtimeFeatureExtractor');

        const features = await this._extract_fbank_features(audio, { center });

        return {
            input_features: features.unsqueeze_(0),
        };
    }
}
