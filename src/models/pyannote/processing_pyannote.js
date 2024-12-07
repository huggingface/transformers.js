import { Processor } from '../../base/processing_utils.js';
import { AutoFeatureExtractor } from '../auto/feature_extraction_auto.js';
import { max, softmax } from '../../utils/maths.js';

export class PyAnnoteProcessor extends Processor {
    static feature_extractor_class = AutoFeatureExtractor

    /**
     * Calls the feature_extractor function with the given audio input.
     * @param {any} audio The audio input to extract features from.
     * @returns {Promise<any>} A Promise that resolves with the extracted features.
     */
    async _call(audio) {
        return await this.feature_extractor(audio)
    }

    /**
     * NOTE: Can return fractional values. `Math.ceil` will ensure correct value.
     * @param {number} samples The number of frames in the audio.
     * @returns {number} The number of frames in the audio.
     */
    samples_to_frames(samples) {
        return ((samples - this.config.offset) / this.config.step);
    }

    /**
     * Post-processes the speaker diarization logits output by the model.
     * @param {import('../../utils/tensor.js').Tensor} logits The speaker diarization logits output by the model.
     * @param {number} num_samples Number of samples in the input audio.
     * @returns {Array<Array<{ id: number, start: number, end: number, confidence: number }>>} The post-processed speaker diarization results.
     */
    post_process_speaker_diarization(logits, num_samples) {
        const ratio = (
            num_samples / this.samples_to_frames(num_samples)
        ) / this.config.sampling_rate;

        const results = [];
        for (const scores of logits.tolist()) {
            const accumulated_segments = [];

            let current_speaker = -1;
            for (let i = 0; i < scores.length; ++i) {
                const probabilities = softmax(scores[i]);
                const [score, id] = max(probabilities);
                const [start, end] = [i, i + 1];

                if (id !== current_speaker) {
                    // Speaker has changed
                    current_speaker = id;
                    accumulated_segments.push({ id, start, end, score });
                } else {
                    // Continue the current segment
                    accumulated_segments.at(-1).end = end;
                    accumulated_segments.at(-1).score += score;
                }
            }

            results.push(accumulated_segments.map(
                // Convert frame-space to time-space
                // and compute the confidence
                ({ id, start, end, score }) => ({
                    id,
                    start: start * ratio,
                    end: end * ratio,
                    confidence: score / (end - start),
                })
            ));
        }
        return results;
    }
}
