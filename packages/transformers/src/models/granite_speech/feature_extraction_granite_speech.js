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

        // Truncate to even number of frames for pair-stacking
        const num_frames = 1 + Math.floor((audio.length - 1) / hop_length);
        const max_num_frames = num_frames - (num_frames % 2);

        const mel = await spectrogram(audio, this.window, n_fft, hop_length, {
            power: 2.0,
            mel_filters: this.mel_filters,
            log_mel: 'log10_max_norm',
            transpose: true, // [time, n_mels]
            max_num_frames,
            do_pad: false,
        });

        // Stack adjacent frame pairs: [time, n_mels] → [1, time/2, 2*n_mels]
        const input_features = mel.view(-1, 2 * n_mels).unsqueeze_(0);

        return { input_features };
    }
}
