import { validate_audio_inputs } from '../../feature_extraction_utils.js';
import { Random } from '../../utils/random.js';
import { ParakeetFeatureExtractor } from '../parakeet/feature_extraction_parakeet.js';

export class CohereAsrFeatureExtractor extends ParakeetFeatureExtractor {
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
     * Extracts features from a given audio waveform.
     * @param {Float32Array|Float64Array} audio The audio data.
     * @returns {Promise<{ input_features: import('../../utils/tensor.js').Tensor; attention_mask: import('../../utils/tensor.js').Tensor; }>}
     */
    async _call(audio) {
        validate_audio_inputs(audio, 'CohereAsrFeatureExtractor');

        // Clone to Float64Array and apply dithering before feature extraction
        const waveform = new Float64Array(audio);
        this._apply_dither(waveform);

        return super._call(waveform);
    }
}
