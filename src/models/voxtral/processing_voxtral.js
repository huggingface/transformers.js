import { AutoFeatureExtractor } from "../auto/feature_extraction_auto.js"
import { AutoTokenizer } from "../../tokenizers.js"
import { Processor } from "../../base/processing_utils.js"
import { cat } from "../../utils/tensor.js";

const AUDIO_TOKEN = "[AUDIO]";
const BEGIN_AUDIO_TOKEN = "[BEGIN_AUDIO]";
const NUM_AUDIO_TOKENS = 375;

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
            const num_audio_tokens = text.split(AUDIO_TOKEN).length - 1;
            if (num_audio_tokens !== audio.length) {
                throw new Error(`The number of audio inputs (${audio.length}) does not match the number of audio tokens in the text (${num_audio_tokens}).`);
            }
            const features = (await Promise.all(
                audio.map((audio_input) => this.feature_extractor(audio_input, kwargs))
            )).map(x => x.input_features);
            audio_inputs["audio_values"] = features.length > 1 ? cat(features, 0) : features[0];

            text = text.replaceAll(AUDIO_TOKEN, BEGIN_AUDIO_TOKEN + AUDIO_TOKEN.repeat(NUM_AUDIO_TOKENS));
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
