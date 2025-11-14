import { Pipeline } from './_base.js';

import { Tensor } from '../utils/tensor.js';
import { RawAudio } from '../utils/audio.js';

import { AutoModel } from '../models.js';

/**
 * @typedef {import('./_base.js').TextAudioPipelineConstructorArgs} TextAudioPipelineConstructorArgs
 * @typedef {import('./_base.js').Disposable} Disposable
 */

/**
 * @typedef {Object} VocoderOptions
 * @property {import('../models.js').PreTrainedModel} [vocoder] The vocoder used by the pipeline (if the model uses one). If not provided, use the default HifiGan vocoder.
 * @typedef {TextAudioPipelineConstructorArgs & VocoderOptions} TextToAudioPipelineConstructorArgs
 */

/**
 * @typedef {Object} TextToAudioOutput
 * @property {Float32Array} audio The generated audio waveform.
 * @property {number} sampling_rate The sampling rate of the generated audio waveform.
 *
 * @typedef {Object} TextToAudioPipelineOptions Parameters specific to text-to-audio pipelines.
 * @property {Tensor|Float32Array|string|URL} [speaker_embeddings=null] The speaker embeddings (if the model requires it).
 *
 * @callback TextToAudioPipelineCallback Generates speech/audio from the inputs.
 * @param {string|string[]} texts The text(s) to generate.
 * @param {TextToAudioPipelineOptions} options Parameters passed to the model generation/forward method.
 * @returns {Promise<TextToAudioOutput>} An object containing the generated audio and sampling rate.
 *
 * @typedef {TextToAudioPipelineConstructorArgs & TextToAudioPipelineCallback & Disposable} TextToAudioPipelineType
 */

/**
 * Text-to-audio generation pipeline using any `AutoModelForTextToWaveform` or `AutoModelForTextToSpectrogram`.
 * This pipeline generates an audio file from an input text and optional other conditional inputs.
 *
 * **Example:** Generate audio from text with `Xenova/speecht5_tts`.
 * ```javascript
 * const synthesizer = await pipeline('text-to-speech', 'Xenova/speecht5_tts', { quantized: false });
 * const speaker_embeddings = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/speaker_embeddings.bin';
 * const out = await synthesizer('Hello, my dog is cute', { speaker_embeddings });
 * // RawAudio {
 * //   audio: Float32Array(26112) [-0.00005657337896991521, 0.00020583874720614403, ...],
 * //   sampling_rate: 16000
 * // }
 * ```
 *
 * You can then save the audio to a .wav file with the `wavefile` package:
 * ```javascript
 * import wavefile from 'wavefile';
 * import fs from 'fs';
 *
 * const wav = new wavefile.WaveFile();
 * wav.fromScratch(1, out.sampling_rate, '32f', out.audio);
 * fs.writeFileSync('out.wav', wav.toBuffer());
 * ```
 *
 * **Example:** Multilingual speech generation with `Xenova/mms-tts-fra`. See [here](https://huggingface.co/models?pipeline_tag=text-to-speech&other=vits&sort=trending) for the full list of available languages (1107).
 * ```javascript
 * const synthesizer = await pipeline('text-to-speech', 'Xenova/mms-tts-fra');
 * const out = await synthesizer('Bonjour');
 * // RawAudio {
 * //   audio: Float32Array(23808) [-0.00037693005288019776, 0.0003325853613205254, ...],
 * //   sampling_rate: 16000
 * // }
 * ```
 */
export class TextToAudioPipeline
    extends /** @type {new (options: TextToAudioPipelineConstructorArgs) => TextToAudioPipelineType} */ (Pipeline)
{
    DEFAULT_VOCODER_ID = 'Xenova/speecht5_hifigan';

    /**
     * Create a new TextToAudioPipeline.
     * @param {TextToAudioPipelineConstructorArgs} options An object used to instantiate the pipeline.
     */
    constructor(options) {
        super(options);

        // TODO: Find a better way for `pipeline` to set the default vocoder
        this.vocoder = options.vocoder ?? null;
    }

    /** @type {TextToAudioPipelineCallback} */
    async _call(text_inputs, { speaker_embeddings = null } = {}) {
        // If this.processor is not set, we are using a `AutoModelForTextToWaveform` model
        if (this.processor) {
            return this._call_text_to_spectrogram(text_inputs, { speaker_embeddings });
        } else {
            return this._call_text_to_waveform(text_inputs);
        }
    }

    async _call_text_to_waveform(text_inputs) {
        // Run tokenization
        const inputs = this.tokenizer(text_inputs, {
            padding: true,
            truncation: true,
        });

        // Generate waveform
        const { waveform } = await this.model(inputs);

        // @ts-expect-error TS2339
        const sampling_rate = this.model.config.sampling_rate;
        return new RawAudio(waveform.data, sampling_rate);
    }

    async _call_text_to_spectrogram(text_inputs, { speaker_embeddings }) {
        // Load vocoder, if not provided
        if (!this.vocoder) {
            console.log('No vocoder specified, using default HifiGan vocoder.');
            this.vocoder = await AutoModel.from_pretrained(this.DEFAULT_VOCODER_ID, { dtype: 'fp32' });
        }

        // Load speaker embeddings as Float32Array from path/URL
        if (typeof speaker_embeddings === 'string' || speaker_embeddings instanceof URL) {
            // Load from URL with fetch
            speaker_embeddings = new Float32Array(await (await fetch(speaker_embeddings)).arrayBuffer());
        }

        if (speaker_embeddings instanceof Float32Array) {
            speaker_embeddings = new Tensor('float32', speaker_embeddings, [1, speaker_embeddings.length]);
        } else if (!(speaker_embeddings instanceof Tensor)) {
            throw new Error('Speaker embeddings must be a `Tensor`, `Float32Array`, `string`, or `URL`.');
        }

        // Run tokenization
        const { input_ids } = this.tokenizer(text_inputs, {
            padding: true,
            truncation: true,
        });

        // NOTE: At this point, we are guaranteed that `speaker_embeddings` is a `Tensor`
        // @ts-ignore
        const { waveform } = await this.model.generate_speech(input_ids, speaker_embeddings, { vocoder: this.vocoder });

        const sampling_rate = this.processor.feature_extractor.config.sampling_rate;
        return new RawAudio(waveform.data, sampling_rate);
    }
}
