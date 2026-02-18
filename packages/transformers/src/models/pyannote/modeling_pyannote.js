import { PreTrainedModel } from '../modeling_utils.js';
import { TokenClassifierOutput } from '../modeling_outputs.js';

export class PyAnnotePreTrainedModel extends PreTrainedModel {}

/**
 * The bare PyAnnote Model transformer outputting raw hidden-states without any specific head on top.
 */
export class PyAnnoteModel extends PyAnnotePreTrainedModel {}

/**
 * PyAnnote Model with a frame classification head on top for tasks like Speaker Diarization.
 *
 * **Example:** Load and run a `PyAnnoteForAudioFrameClassification` for speaker diarization.
 *
 * ```javascript
 * import { AutoProcessor, AutoModelForAudioFrameClassification, read_audio } from '@huggingface/transformers';
 *
 * // Load model and processor
 * const model_id = 'onnx-community/pyannote-segmentation-3.0';
 * const model = await AutoModelForAudioFrameClassification.from_pretrained(model_id);
 * const processor = await AutoProcessor.from_pretrained(model_id);
 *
 * // Read and preprocess audio
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/mlk.wav';
 * const audio = await read_audio(url, processor.feature_extractor.config.sampling_rate);
 * const inputs = await processor(audio);
 *
 * // Run model with inputs
 * const { logits } = await model(inputs);
 * // {
 * //   logits: Tensor {
 * //     dims: [ 1, 767, 7 ],  // [batch_size, num_frames, num_classes]
 * //     type: 'float32',
 * //     data: Float32Array(5369) [ ... ],
 * //     size: 5369
 * //   }
 * // }
 *
 * const result = processor.post_process_speaker_diarization(logits, audio.length);
 * // [
 * //   [
 * //     { id: 0, start: 0, end: 1.0512535626298245, confidence: 0.8220156481664611 },
 * //     { id: 2, start: 1.0512535626298245, end: 2.3398869619825127, confidence: 0.9008811707860472 },
 * //     ...
 * //   ]
 * // ]
 *
 * // Display result
 * console.table(result[0], ['start', 'end', 'id', 'confidence']);
 * // ┌─────────┬────────────────────┬────────────────────┬────┬─────────────────────┐
 * // │ (index) │ start              │ end                │ id │ confidence          │
 * // ├─────────┼────────────────────┼────────────────────┼────┼─────────────────────┤
 * // │ 0       │ 0                  │ 1.0512535626298245 │ 0  │ 0.8220156481664611  │
 * // │ 1       │ 1.0512535626298245 │ 2.3398869619825127 │ 2  │ 0.9008811707860472  │
 * // │ 2       │ 2.3398869619825127 │ 3.5946089560890773 │ 0  │ 0.7521651315796233  │
 * // │ 3       │ 3.5946089560890773 │ 4.578039708226655  │ 2  │ 0.8491978128022479  │
 * // │ 4       │ 4.578039708226655  │ 4.594995410849717  │ 0  │ 0.2935352600416393  │
 * // │ 5       │ 4.594995410849717  │ 6.121008646925269  │ 3  │ 0.6788051309866024  │
 * // │ 6       │ 6.121008646925269  │ 6.256654267909762  │ 0  │ 0.37125512393851134 │
 * // │ 7       │ 6.256654267909762  │ 8.630452635138397  │ 2  │ 0.7467035186353542  │
 * // │ 8       │ 8.630452635138397  │ 10.088643060721703 │ 0  │ 0.7689364814666032  │
 * // │ 9       │ 10.088643060721703 │ 12.58113134631177  │ 2  │ 0.9123324509131324  │
 * // │ 10      │ 12.58113134631177  │ 13.005023911888312 │ 0  │ 0.4828358177572041  │
 * // └─────────┴────────────────────┴────────────────────┴────┴─────────────────────┘
 * ```
 */
export class PyAnnoteForAudioFrameClassification extends PyAnnotePreTrainedModel {
    /**
     * Calls the model on new inputs.
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<TokenClassifierOutput>} An object containing the model's output logits for sequence classification.
     */
    async _call(model_inputs) {
        return new TokenClassifierOutput(await super._call(model_inputs));
    }
}
