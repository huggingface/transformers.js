import { FeatureExtractor, validate_audio_inputs } from '../../feature_extraction_utils.js';
import { Tensor } from '../../utils/tensor.js';
import { mel_filter_bank, spectrogram, window_function } from '../../utils/audio.js';
import { Random } from '../../utils/random.js';

const EPSILON = 1e-5;

export class CohereAsrFeatureExtractor extends FeatureExtractor {
    constructor(config) {
        super(config);

        this.config.mel_filters ??= mel_filter_bank(
            Math.floor(1 + this.config.n_fft / 2), // num_frequency_bins (257)
            this.config.feature_size, // num_mel_filters (128)
            0.0, // min_frequency
            this.config.sampling_rate / 2, // max_frequency (8000)
            this.config.sampling_rate, // sampling_rate (16000)
            'slaney', // norm
            'slaney', // mel_scale
        );

        const window = window_function(this.config.win_length, 'hann', {
            periodic: false,
        });

        this.window = new Float64Array(this.config.n_fft);
        const offset = Math.floor((this.config.n_fft - this.config.win_length) / 2);
        this.window.set(window, offset);
    }

    /**
     * Apply deterministic dithering seeded by the waveform length.
     * @param {Float64Array} waveform
     * @returns {Float64Array} The dithered waveform (mutated in-place).
     */
    _apply_dither(waveform) {
        const dither = this.config.dither ?? 0;
        if (dither <= 0) return waveform;

        const rng = new Random(waveform.length);
        for (let i = 0; i < waveform.length; ++i) {
            waveform[i] += dither * rng.gauss();
        }
        return waveform;
    }

    /**
     * Split audio into chunks at energy-based boundaries for long audio.
     * @param {Float32Array|Float64Array} audio The raw audio waveform.
     * @returns {(Float32Array|Float64Array)[]} Array of audio chunks.
     */
    split_audio(audio) {
        const max_audio_clip_s = this.config.max_audio_clip_s ?? 35.0;
        const overlap_chunk_second = this.config.overlap_chunk_second ?? 5.0;
        const min_energy_window_samples = this.config.min_energy_window_samples ?? 1600;
        const sampling_rate = this.config.sampling_rate;

        const chunk_size = Math.max(1, Math.round(max_audio_clip_s * sampling_rate));
        const boundary_context_size = Math.max(1, Math.round(overlap_chunk_second * sampling_rate));

        if (audio.length <= chunk_size) {
            return [audio];
        }

        const chunks = [];
        let idx = 0;
        const total_samples = audio.length;

        while (idx < total_samples) {
            if (idx + chunk_size >= total_samples) {
                chunks.push(audio.slice(idx, total_samples));
                break;
            }

            // Search for quietest point near the chunk boundary
            const search_start = Math.max(idx, idx + chunk_size - boundary_context_size);
            const search_end = Math.min(idx + chunk_size, total_samples);

            let split_point;
            if (search_end <= search_start) {
                split_point = idx + chunk_size;
            } else {
                split_point = this._find_split_point_energy(audio, search_start, search_end, min_energy_window_samples);
            }

            split_point = Math.max(idx + 1, Math.min(split_point, total_samples));
            chunks.push(audio.slice(idx, split_point));
            idx = split_point;
        }

        return chunks;
    }

    /**
     * Find the quietest point (minimum energy) within a segment of audio.
     * @param {Float32Array|Float64Array} waveform
     * @param {number} start_idx
     * @param {number} end_idx
     * @param {number} window_size
     * @returns {number} Index of the quietest point.
     */
    _find_split_point_energy(waveform, start_idx, end_idx, window_size) {
        const segment_len = end_idx - start_idx;
        if (segment_len <= window_size) {
            return Math.floor((start_idx + end_idx) / 2);
        }

        let min_energy = Infinity;
        let quietest_idx = start_idx;
        const upper = segment_len - window_size;

        for (let i = 0; i <= upper; i += window_size) {
            let energy = 0;
            for (let j = 0; j < window_size; ++j) {
                const val = waveform[start_idx + i + j];
                energy += val * val;
            }
            energy = Math.sqrt(energy / window_size);

            if (energy < min_energy) {
                min_energy = energy;
                quietest_idx = start_idx + i;
            }
        }

        return quietest_idx;
    }

    /**
     * Computes the log-Mel spectrogram of the provided audio waveform.
     * @param {Float64Array} waveform The audio waveform to process.
     * @returns {Promise<Tensor>} The log-Mel spectrogram tensor.
     */
    async _extract_fbank_features(waveform) {
        // Apply preemphasis
        const preemphasis = this.config.preemphasis;
        for (let j = waveform.length - 1; j >= 1; --j) {
            waveform[j] -= preemphasis * waveform[j - 1];
        }

        const features = await spectrogram(
            waveform,
            this.window, // window
            this.window.length, // frame_length
            this.config.hop_length, // hop_length
            {
                fft_length: this.config.n_fft,
                power: 2.0,
                mel_filters: this.config.mel_filters,
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
     * Extracts features from a given audio waveform.
     * @param {Float32Array|Float64Array} audio The audio data.
     * @returns {Promise<{ input_features: Tensor; attention_mask: Tensor; }>}
     */
    async _call(audio) {
        validate_audio_inputs(audio, 'CohereAsrFeatureExtractor');

        // Clone to Float64Array and apply dithering
        const waveform = new Float64Array(audio);
        this._apply_dither(waveform);

        const features = await this._extract_fbank_features(waveform);

        const features_length = Math.floor(
            (audio.length + Math.floor(this.config.n_fft / 2) * 2 - this.config.n_fft) / this.config.hop_length,
        );

        const features_data = /** @type {Float32Array} */ (features.data);
        features_data.fill(0, features_length * features.dims[1]);

        // Normalize mel features per-feature, ignoring padding
        const [num_frames, num_features] = features.dims;
        const sum = new Float64Array(num_features);
        const sum_sq = new Float64Array(num_features);

        for (let i = 0; i < features_length; ++i) {
            const offset = i * num_features;
            for (let j = 0; j < num_features; ++j) {
                const val = features_data[offset + j];
                sum[j] += val;
                sum_sq[j] += val * val;
            }
        }

        const divisor = features_length > 1 ? features_length - 1 : 1;
        for (let j = 0; j < num_features; ++j) {
            const mean = sum[j] / features_length;
            const variance = (sum_sq[j] - features_length * mean * mean) / divisor;
            const std = Math.sqrt(variance) + EPSILON;
            const inv_std = 1 / std;

            for (let i = 0; i < features_length; ++i) {
                const index = i * num_features + j;
                features_data[index] = (features_data[index] - mean) * inv_std;
            }
        }

        const mask_data = new BigInt64Array(num_frames);
        mask_data.fill(1n, 0, features_length);

        return {
            input_features: features.unsqueeze_(0),
            attention_mask: new Tensor('int64', mask_data, [1, num_frames]),
        };
    }
}
