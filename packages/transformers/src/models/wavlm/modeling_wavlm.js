import { PreTrainedModel } from '../modeling_utils.js';
import { CausalLMOutput, SequenceClassifierOutput, TokenClassifierOutput, ModelOutput } from '../modeling_outputs.js';
import { Tensor } from '../../utils/tensor.js';

/**
 * Base class for outputs of XVector models.
 */
export class XVectorOutput extends ModelOutput {
    /**
     * @param {Object} output The output of the model.
     * @param {Tensor} output.logits Classification hidden states before AMSoftmax, of shape `(batch_size, config.xvector_output_dim)`.
     * @param {Tensor} output.embeddings Utterance embeddings used for vector similarity-based retrieval, of shape `(batch_size, config.xvector_output_dim)`.
     */
    constructor({ logits, embeddings }) {
        super();
        this.logits = logits;
        this.embeddings = embeddings;
    }
}

/**
 * An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained models.
 */
export class WavLMPreTrainedModel extends PreTrainedModel {}

/**
 * The bare WavLM Model transformer outputting raw hidden-states without any specific head on top.
 *
 * **Example:** Load and run a `WavLMModel` for feature extraction.
 *
 * ```javascript
 * import { AutoProcessor, AutoModel, read_audio } from '@huggingface/transformers';
 *
 * // Read and preprocess audio
 * const processor = await AutoProcessor.from_pretrained('Xenova/wavlm-base');
 * const audio = await read_audio('https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav', 16000);
 * const inputs = await processor(audio);
 *
 * // Run model with inputs
 * const model = await AutoModel.from_pretrained('Xenova/wavlm-base');
 * const output = await model(inputs);
 * // {
 * //   last_hidden_state: Tensor {
 * //     dims: [ 1, 549, 768 ],
 * //     type: 'float32',
 * //     data: Float32Array(421632) [-0.349443256855011, -0.39341306686401367,  0.022836603224277496, ...],
 * //     size: 421632
 * //   }
 * // }
 * ```
 */
export class WavLMModel extends WavLMPreTrainedModel {}

/**
 * WavLM Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).
 */
export class WavLMForCTC extends WavLMPreTrainedModel {
    /**
     * @param {Object} model_inputs
     * @param {Tensor} model_inputs.input_values Float values of input raw speech waveform.
     * @param {Tensor} model_inputs.attention_mask Mask to avoid performing convolution and attention on padding token indices. Mask values selected in [0, 1]
     */
    async _call(model_inputs) {
        return new CausalLMOutput(await super._call(model_inputs));
    }
}

/**
 * WavLM Model with a sequence classification head on top (a linear layer over the pooled output).
 */
export class WavLMForSequenceClassification extends WavLMPreTrainedModel {
    /**
     * Calls the model on new inputs.
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<SequenceClassifierOutput>} An object containing the model's output logits for sequence classification.
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}

/**
 * WavLM Model with an XVector feature extraction head on top for tasks like Speaker Verification.
 *
 * **Example:** Extract speaker embeddings with `WavLMForXVector`.
 * ```javascript
 * import { AutoProcessor, AutoModel, read_audio } from '@huggingface/transformers';
 *
 * // Read and preprocess audio
 * const processor = await AutoProcessor.from_pretrained('Xenova/wavlm-base-plus-sv');
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav';
 * const audio = await read_audio(url, 16000);
 * const inputs = await processor(audio);
 *
 * // Run model with inputs
 * const model = await AutoModel.from_pretrained('Xenova/wavlm-base-plus-sv');
 * const outputs = await model(inputs);
 * // {
 * //   logits: Tensor {
 * //     dims: [ 1, 512 ],
 * //     type: 'float32',
 * //     data: Float32Array(512) [0.5847219228744507, ...],
 * //     size: 512
 * //   },
 * //   embeddings: Tensor {
 * //     dims: [ 1, 512 ],
 * //     type: 'float32',
 * //     data: Float32Array(512) [-0.09079201519489288, ...],
 * //     size: 512
 * //   }
 * // }
 * ```
 */
export class WavLMForXVector extends WavLMPreTrainedModel {
    /**
     * Calls the model on new inputs.
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<XVectorOutput>} An object containing the model's output logits and speaker embeddings.
     */
    async _call(model_inputs) {
        return new XVectorOutput(await super._call(model_inputs));
    }
}

/**
 * WavLM Model with a frame classification head on top for tasks like Speaker Diarization.
 *
 * **Example:** Perform speaker diarization with `WavLMForAudioFrameClassification`.
 * ```javascript
 * import { AutoProcessor, AutoModelForAudioFrameClassification, read_audio } from '@huggingface/transformers';
 *
 * // Read and preprocess audio
 * const processor = await AutoProcessor.from_pretrained('Xenova/wavlm-base-plus-sd');
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav';
 * const audio = await read_audio(url, 16000);
 * const inputs = await processor(audio);
 *
 * // Run model with inputs
 * const model = await AutoModelForAudioFrameClassification.from_pretrained('Xenova/wavlm-base-plus-sd');
 * const { logits } = await model(inputs);
 * // {
 * //   logits: Tensor {
 * //     dims: [ 1, 549, 2 ],  // [batch_size, num_frames, num_speakers]
 * //     type: 'float32',
 * //     data: Float32Array(1098) [-3.5301010608673096, ...],
 * //     size: 1098
 * //   }
 * // }
 *
 * const labels = logits[0].sigmoid().tolist().map(
 *     frames => frames.map(speaker => speaker > 0.5 ? 1 : 0)
 * );
 * console.log(labels); // labels is a one-hot array of shape (num_frames, num_speakers)
 * // [
 * //     [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
 * //     [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
 * //     [0, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1],
 * //     ...
 * // ]
 * ```
 */
export class WavLMForAudioFrameClassification extends WavLMPreTrainedModel {
    /**
     * Calls the model on new inputs.
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<TokenClassifierOutput>} An object containing the model's output logits for sequence classification.
     */
    async _call(model_inputs) {
        return new TokenClassifierOutput(await super._call(model_inputs));
    }
}
