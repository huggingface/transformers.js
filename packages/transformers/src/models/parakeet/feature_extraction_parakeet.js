import { FeatureExtractor, validate_audio_inputs } from '../../feature_extraction_utils.js';
import { Tensor, full, matmul, conv1d, add, mul, pow, pad } from '../../utils/tensor.js';
import { mel_filter_bank, spectrogram, window_function, spectrogram_kernel } from '../../utils/audio.js';

const EPSILON = 1e-5;

export class ParakeetFeatureExtractor extends FeatureExtractor {
    constructor(config) {
        super(config);

        // Prefer given `mel_filters` from preprocessor_config.json, or calculate them if they don't exist.
        this.config.mel_filters ??= mel_filter_bank(
            Math.floor(1 + this.config.n_fft / 2), // num_frequency_bins
            this.config.feature_size, // num_mel_filters
            0.0, // min_frequency
            this.config.sampling_rate / 2, // max_frequency
            this.config.sampling_rate, // sampling_rate
            'slaney', // norm
            'slaney', // mel_scale
        );

        const window = window_function(this.config.win_length, 'hann', {
            periodic: false,
        });

        this.window = new Float64Array(this.config.n_fft);
        const offset = Math.floor((this.config.n_fft - this.config.win_length) / 2);
        this.window.set(window, offset);

        this.spectrogram_kernel = spectrogram_kernel(this.window, this.config.hop_length, {
            fft_length: this.config.n_fft,
        });

        // Preemphasis filter y[j] = x[j] - preemphasis * x[j - 1], as a convolution kernel.
        this.preemphasis_kernel = new Tensor('float32', Float32Array.of(-this.config.preemphasis, 1), [1, 1, 2]);
    }

    /**
     * Computes the log-Mel spectrogram of the provided audio waveform.
     * @param {Float32Array|Float64Array} waveform The audio waveform to process.
     * @returns {Promise<Tensor>} An object containing the log-Mel spectrogram data as a Float32Array and its dimensions as an array of numbers.
     */
    async _extract_fbank_features(waveform) {
        // Parakeet uses a custom preemphasis strategy: Apply preemphasis to entire waveform at once
        // (with a zero before the first sample, so that y[0] = x[0])
        const signal = new Tensor('float32', new Float32Array(waveform), [1, 1, waveform.length]);
        const emphasized = await conv1d(await pad(signal, [1, 0]), this.preemphasis_kernel);

        const features = await spectrogram(
            /** @type {Float32Array} */ (emphasized.data),
            this.window, // window
            this.window.length, // frame_length
            this.config.hop_length, // hop_length
            {
                fft_length: this.config.n_fft,
                power: 2.0,
                mel_filters: this.config.mel_filters,
                kernel: this.spectrogram_kernel,
                log_mel: 'log',
                mel_floor: -Infinity,
                pad_mode: 'constant',
                center: true,

                // Custom
                transpose: true,
                mel_offset: 2 ** -24,
            },
        );

        return features;
    }

    /**
     * Asynchronously extracts features from a given audio using the provided configuration.
     * @param {Float32Array|Float64Array} audio The audio data as a Float32Array/Float64Array.
     * @returns {Promise<{ input_features: Tensor; attention_mask: Tensor; }>} A Promise resolving to an object containing the extracted input features as a Tensor.
     */
    async _call(audio) {
        validate_audio_inputs(audio, 'ParakeetFeatureExtractor');

        const mel = await this._extract_fbank_features(audio);
        const [num_frames, num_features] = mel.dims;

        const features_length = Math.min(
            num_frames,
            Math.floor(
                (audio.length + Math.floor(this.config.n_fft / 2) * 2 - this.config.n_fft) / this.config.hop_length,
            ),
        );

        // normalize mel features over the valid frames ((x - mean) / (std + EPSILON), with ddof=1); padding frames are zero.
        // Means over frames are computed as [1, frames] @ [frames, mel] products.
        let features;
        if (features_length > 0) {
            const valid = new Tensor(
                'float32',
                /** @type {Float32Array} */ (mel.data).subarray(0, features_length * num_features),
                [features_length, num_features],
            );
            const neg_mean = await matmul(full([1, features_length], -1 / features_length), valid);
            const centered = await add(valid, neg_mean);
            const divisor = features_length > 1 ? features_length - 1 : 1;
            const variance = await matmul(full([1, features_length], 1 / divisor), await pow(centered, 2));
            const inv_std = await pow(await add(await pow(variance, 0.5), EPSILON), -1);
            features = await mul(centered, inv_std);
            if (features_length < num_frames) {
                features = await pad(features, [0, 0, 0, num_frames - features_length]);
            }
        } else {
            features = full([num_frames, num_features], 0);
        }

        const mask_data = new BigInt64Array(num_frames);
        mask_data.fill(1n, 0, features_length);

        return {
            input_features: features.unsqueeze_(0),
            attention_mask: new Tensor('int64', mask_data, [1, num_frames]),
        };
    }
}
