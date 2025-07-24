import { AutoFeatureExtractor } from "../auto/feature_extraction_auto.js"
import { AutoTokenizer } from "../../tokenizers.js"
import { Processor } from "../../base/processing_utils.js"
import { cat } from "../../utils/tensor.js";

const AUDIO_TOKEN = "[AUDIO]";
const BEGIN_AUDIO_TOKEN = "[BEGIN_AUDIO]";
const NUM_AUDIO_TOKENS = 375;

/**
 * Helper function to split audio into non-overlapping chunks of n_samples
 * @param {Float32Array} audio 
 * @param {number} n_samples 
 * @returns {Float32Array[]}
 */
function chunk(audio, n_samples) {
    const chunks = [];
    for (let i = 0; i < audio.length; i += n_samples) {
        chunks.push(audio.subarray(i, Math.min(i + n_samples, audio.length)));
    }
    return chunks;
}

/**
 * Represents a VoxtralProcessor that extracts features from an audio input.
 */
export class VoxtralProcessor extends Processor {
    static tokenizer_class = AutoTokenizer
    static feature_extractor_class = AutoFeatureExtractor
    static uses_processor_config = false;

    /**
     * @param {string} text The text input to process.
     * @param {Float32Array|Float32Array[]} audio The audio input(s) to process.
     */
    async _call(text, audio = null, kwargs = {}) {
        if (Array.isArray(text)) {
            throw new Error("Batched inputs are not supported yet.");
        }

        const audio_inputs = {};
        if (audio) {
            if (!text.includes(AUDIO_TOKEN)) {
                throw new Error(`The input text does not contain the audio token ${AUDIO_TOKEN}.`);
            }
            if (!Array.isArray(audio)) {
                audio = [audio];
            }
            const text_parts = text.split(AUDIO_TOKEN);
            const num_audio_tokens = text_parts.length - 1;
            if (num_audio_tokens !== audio.length) {
                throw new Error(`The number of audio inputs (${audio.length}) does not match the number of audio tokens in the text (${num_audio_tokens}).`);
            }

            const n_samples = this.feature_extractor.config.n_samples;

            // Split each audio input into chunks and keep track of chunk counts
            const audio_chunks = audio.map(a => chunk(a, n_samples));
            const chunk_counts = audio_chunks.map(chunks => chunks.length);

            // Flatten all chunks for feature extraction
            const all_chunks = audio_chunks.flat();
            const features = (await Promise.all(
                all_chunks.map((audio_input) => this.feature_extractor(audio_input, kwargs))
            )).map(x => x.input_features);

            audio_inputs["audio_values"] = features.length > 1 ? cat(features, 0) : features[0];

            // Replace text tokens for each audio input, expanding for chunk count
            let new_text = text_parts[0];
            for (let i = 0; i < chunk_counts.length; ++i) {
                new_text += BEGIN_AUDIO_TOKEN;
                for (let j = 0; j < chunk_counts[i]; ++j) {
                    new_text += AUDIO_TOKEN.repeat(NUM_AUDIO_TOKENS);
                }
                new_text += text_parts[i + 1];
            }
            text = new_text;
        }

        const text_inputs = this.tokenizer(text, {
            add_special_tokens: false,
            ...kwargs,
        });

        return {
            ...text_inputs,
            ...audio_inputs,
        }
    }
}
