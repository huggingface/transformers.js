import { Pipeline, prepareAudios } from './_base.js';

import { Tensor, topk } from '../utils/tensor.js';
import { softmax } from '../utils/maths.js';

/**
 * @typedef {import('./_base.js').AudioPipelineConstructorArgs} AudioPipelineConstructorArgs
 * @typedef {import('./_base.js').Disposable} Disposable
 * @typedef {import('./_base.js').AudioInput} AudioInput
 */

/**
 * @typedef {Object} AudioClassificationSingle
 * @property {string} label The label predicted.
 * @property {number} score The corresponding probability.
 * @typedef {AudioClassificationSingle[]} AudioClassificationOutput
 *
 * @typedef {Object} AudioClassificationPipelineOptions Parameters specific to audio classification pipelines.
 * @property {number} [top_k=5] The number of top labels that will be returned by the pipeline.
 * If the provided number is `null` or higher than the number of labels available in the model configuration,
 * it will default to the number of labels.
 *
 * @callback AudioClassificationPipelineCallbackSingle Classify a single audio input.
 * @param {AudioInput} audio The input audio file(s) to be classified. The input is either:
 * - `string` or `URL` that is the filename/URL of the audio file, the file will be read at the processor's sampling rate
 * to get the waveform using the [`AudioContext`](https://developer.mozilla.org/en-US/docs/Web/API/AudioContext) API.
 * If `AudioContext` is not available, you should pass the raw waveform in as a Float32Array of shape `(n, )`.
 * - `Float32Array` or `Float64Array` of shape `(n, )`, representing the raw audio at the correct sampling rate (no further check will be done).
 * @param {AudioClassificationPipelineOptions} [options] The options to use for audio classification.
 * @returns {Promise<AudioClassificationOutput>} An array containing the predicted labels and scores.
 *
 * @callback AudioClassificationPipelineCallbackBatch Classify multiple audio inputs.
 * @param {AudioInput[]} audio The audio inputs to be classified.
 * @param {AudioClassificationPipelineOptions} [options] The options to use for audio classification.
 * @returns {Promise<AudioClassificationOutput[]>} An array of prediction arrays corresponding to each input.
 *
 * @typedef {AudioClassificationPipelineCallbackSingle & AudioClassificationPipelineCallbackBatch} AudioClassificationPipelineCallback
 *
 * @typedef {AudioPipelineConstructorArgs & AudioClassificationPipelineCallback & Disposable} AudioClassificationPipelineType
 */

/**
 * Audio classification pipeline using any `AutoModelForAudioClassification`.
 * This pipeline predicts the class of a raw waveform or an audio file.
 *
 * **Example:** Perform audio classification with `Xenova/wav2vec2-large-xlsr-53-gender-recognition-librispeech`.
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const classifier = await pipeline('audio-classification', 'Xenova/wav2vec2-large-xlsr-53-gender-recognition-librispeech');
 * const audio = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav';
 * const output = await classifier(audio);
 * // [
 * //   { label: 'male', score: 0.9981542229652405 },
 * //   { label: 'female', score: 0.001845747814513743 }
 * // ]
 * ```
 *
 * **Example:** Perform audio classification with `Xenova/ast-finetuned-audioset-10-10-0.4593` and return top 4 results.
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const classifier = await pipeline('audio-classification', 'Xenova/ast-finetuned-audioset-10-10-0.4593');
 * const audio = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cat_meow.wav';
 * const output = await classifier(audio, { top_k: 4 });
 * // [
 * //   { label: 'Meow', score: 0.5617874264717102 },
 * //   { label: 'Cat', score: 0.22365376353263855 },
 * //   { label: 'Domestic animals, pets', score: 0.1141069084405899 },
 * //   { label: 'Animal', score: 0.08985692262649536 },
 * // ]
 * ```
 */
export class AudioClassificationPipeline
    extends /** @type {new (options: AudioPipelineConstructorArgs) => AudioClassificationPipelineType} */ (Pipeline)
{
    async _call(audio, { top_k = 5 } = {}) {
        const sampling_rate = this.processor.feature_extractor.config.sampling_rate;
        const preparedAudios = await prepareAudios(audio, sampling_rate);

        // @ts-expect-error TS2339
        const id2label = this.model.config.id2label;

        const toReturn = [];
        for (const aud of preparedAudios) {
            const inputs = await this.processor(aud);
            const output = await this.model(inputs);
            const logits = output.logits[0];

            const scores = await topk(new Tensor('float32', softmax(logits.data), logits.dims), top_k);

            const values = scores[0].tolist();
            const indices = scores[1].tolist();

            const vals = indices.map((x, i) => ({
                label: /** @type {string} */ (id2label ? id2label[x] : `LABEL_${x}`),
                score: /** @type {number} */ (values[i]),
            }));

            toReturn.push(vals);
        }
        return Array.isArray(audio) ? toReturn : toReturn[0];
    }
}
