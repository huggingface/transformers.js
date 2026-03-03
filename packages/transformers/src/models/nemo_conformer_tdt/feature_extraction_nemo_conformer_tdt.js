import { FeatureExtractor, validate_audio_inputs } from '../../feature_extraction_utils.js';
import { Tensor } from '../../utils/tensor.js';
import { mel_filter_bank, spectrogram, window_function } from '../../utils/audio.js';
import { logger } from '../../utils/logger.js';
import { FeatureLRUCache, createAudioCacheKey } from './transducer_cache.js';
import { computeTemporalDeltas } from './transducer_deltas.js';

const EPSILON = 1e-5;

/**
 * Feature extractor for Nemo Conformer TDT models.
 *
 * Mirrors NeMo-style log-mel extraction used by Parakeet with configurable
 * `feature_size` (e.g. 80 or 128 mel bins via `preprocessor_config.json`).
 */
export class NemoConformerTDTFeatureExtractor extends FeatureExtractor {
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

        // Optional feature-level cache and delta/delta-delta post-processing.
        this.use_feature_cache = this.config.use_feature_cache ?? false;
        this.delta_order = this.config.delta_order ?? 0;
        this.delta_window = this.config.delta_window ?? 2;
        this.delta_concatenate = this.config.delta_concatenate ?? true;

        if (![0, 1, 2].includes(this.delta_order)) {
            throw new Error(
                `NemoConformerTDTFeatureExtractor expected delta_order in {0,1,2}, got ${this.delta_order}.`,
            );
        }
        if (!Number.isInteger(this.delta_window) || this.delta_window < 1) {
            throw new Error(
                `NemoConformerTDTFeatureExtractor expected \`delta_window\` as a positive integer, got ${this.delta_window}.`,
            );
        }
        if (this.delta_order > 0 && !this.delta_concatenate) {
            logger.warn(
                'NemoConformerTDTFeatureExtractor: `delta_concatenate=false` is set. ' +
                    '`input_features` will remain base features and deltas are returned in extra fields.',
            );
        }

        this.feature_cache = this.use_feature_cache
            ? new FeatureLRUCache({
                  max_entries: this.config.feature_cache_max_entries ?? 128,
                  max_size_mb: this.config.feature_cache_max_size_mb ?? 64,
              })
            : null;
    }

    /**
     * Computes the log-Mel spectrogram of the provided audio waveform.
     * @param {Float32Array|Float64Array} waveform The audio waveform to process.
     * @returns {Promise<Tensor>} An object containing the log-Mel spectrogram data as a Float32Array and its dimensions as an array of numbers.
     */
    async _extract_fbank_features(waveform) {
        // Parakeet uses a custom preemphasis strategy: Apply preemphasis to entire waveform at once
        const preemphasis = this.config.preemphasis ?? 0;
        if (!Number.isFinite(preemphasis) || preemphasis < 0 || preemphasis >= 1) {
            throw new Error(
                `NemoConformerTDTFeatureExtractor expected \`preemphasis\` in [0, 1), got ${this.config.preemphasis}.`,
            );
        }
        waveform = new Float64Array(waveform); // Clone to avoid destructive changes
        if (preemphasis !== 0) {
            for (let j = waveform.length - 1; j >= 1; --j) {
                waveform[j] -= preemphasis * waveform[j - 1];
            }
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
     * Asynchronously extracts features from a given audio using the provided configuration.
     * @param {Float32Array|Float64Array} audio The audio data as a Float32Array/Float64Array.
     * @returns {Promise<{
     *  input_features: Tensor;
     *  attention_mask: Tensor;
     *  delta_features?: Tensor;
     *  delta_delta_features?: Tensor;
     * }>} A Promise resolving to an object containing extracted model inputs.
     *      When cache is enabled, tensor instances are shared with cached entries.
     *      Do not mutate or dispose returned tensors unless cache is disabled/cleared.
     */
    async _call(audio) {
        validate_audio_inputs(audio, 'NemoConformerTDTFeatureExtractor');

        if (this.feature_cache) {
            const key = `${createAudioCacheKey(audio, this.config.sampling_rate)}:${this.delta_order}:${this.delta_window}:${this.delta_concatenate}`;
            const cached = this.feature_cache.get(key);
            if (cached) {
                return { ...cached };
            }

            const extracted = await this._extract(audio);
            this.feature_cache.set(key, extracted);
            return extracted;
        }

        return await this._extract(audio);
    }

    async _extract(audio) {
        const features = await this._extract_fbank_features(audio);

        const [num_frames, num_features] = features.dims;
        const raw_features_length = Math.floor(
            (audio.length + Math.floor(this.config.n_fft / 2) * 2 - this.config.n_fft) / this.config.hop_length,
        );
        // Clamp to [0, num_frames] to avoid a negative fill offset for very short clips.
        const features_length = Math.max(0, Math.min(num_frames, raw_features_length));

        const features_data = /** @type {Float32Array} */ (features.data);
        features_data.fill(0, features_length * num_features);

        // normalize mel features, ignoring padding
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

        // Skip normalization for empty/very short audio to avoid NaN from divide-by-zero.
        if (features_length > 0) {
            // Calculate mean and standard deviation, then normalize
            const divisor = features_length > 1 ? features_length - 1 : 1;
            for (let j = 0; j < num_features; ++j) {
                const mean = sum[j] / features_length;
                const variance = (sum_sq[j] - features_length * mean * mean) / divisor;
                const std = Math.sqrt(Math.max(variance, 0)) + EPSILON;
                const inv_std = 1 / std;

                for (let i = 0; i < features_length; ++i) {
                    const index = i * num_features + j;
                    features_data[index] = (features_data[index] - mean) * inv_std;
                }
            }
        }

        const mask_data = new BigInt64Array(num_frames);
        mask_data.fill(1n, 0, features_length);

        let input_features = features.unsqueeze_(0);
        const attention_mask = new Tensor('int64', mask_data, [1, num_frames]);

        const result = {
            input_features,
            attention_mask,
        };

        if (this.delta_order > 0) {
            const delta_result = computeTemporalDeltas(input_features, {
                order: this.delta_order,
                window: this.delta_window,
                concatenate: this.delta_concatenate,
            });
            if (delta_result instanceof Tensor) {
                input_features.dispose();
                input_features = delta_result;
                result.input_features = input_features;
            } else {
                result.delta_features = delta_result.delta;
                if (delta_result.delta_delta) {
                    result.delta_delta_features = delta_result.delta_delta;
                }
            }
        }

        return result;
    }

    clear_cache() {
        this.feature_cache?.clear();
    }

    get_cache_stats() {
        return this.feature_cache?.stats() ?? null;
    }
}
