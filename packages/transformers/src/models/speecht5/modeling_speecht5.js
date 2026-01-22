import { PreTrainedModel, encoder_forward, boolTensor } from '../modeling_utils.js';
import { sessionRun } from '../session.js';
import { Tensor, cat } from '../../utils/tensor.js';

/**
 * An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained models.
 */
export class SpeechT5PreTrainedModel extends PreTrainedModel {}

/**
 * The bare SpeechT5 Encoder-Decoder Model outputting raw hidden-states without any specific pre- or post-nets.
 */
export class SpeechT5Model extends SpeechT5PreTrainedModel {}

/**
 * SpeechT5 Model with a speech encoder and a text decoder.
 *
 * **Example:** Generate speech from text with `SpeechT5ForSpeechToText`.
 * ```javascript
 * import { AutoTokenizer, AutoProcessor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, Tensor } from '@huggingface/transformers';
 *
 * // Load the tokenizer and processor
 * const tokenizer = await AutoTokenizer.from_pretrained('Xenova/speecht5_tts');
 * const processor = await AutoProcessor.from_pretrained('Xenova/speecht5_tts');
 *
 * // Load the models
 * // NOTE: We use the full-precision versions as they are more accurate
 * const model = await SpeechT5ForTextToSpeech.from_pretrained('Xenova/speecht5_tts', { dtype: 'fp32' });
 * const vocoder = await SpeechT5HifiGan.from_pretrained('Xenova/speecht5_hifigan', { dtype: 'fp32' });
 *
 * // Load speaker embeddings from URL
 * const speaker_embeddings_data = new Float32Array(
 *   await (await fetch('https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/speaker_embeddings.bin')).arrayBuffer()
 * );
 * const speaker_embeddings = new Tensor(
 *   'float32',
 *   speaker_embeddings_data,
 *   [1, speaker_embeddings_data.length]
 * )
 *
 * // Run tokenization
 * const { input_ids } = tokenizer('Hello, my dog is cute');
 *
 * // Generate waveform
 * const { waveform } = await model.generate_speech(input_ids, speaker_embeddings, { vocoder });
 * console.log(waveform)
 * // Tensor {
 * //   dims: [ 26112 ],
 * //   type: 'float32',
 * //   size: 26112,
 * //   data: Float32Array(26112) [ -0.00043630177970044315, -0.00018082228780258447, ... ],
 * // }
 * ```
 */
export class SpeechT5ForSpeechToText extends SpeechT5PreTrainedModel {}

/**
 * SpeechT5 Model with a text encoder and a speech decoder.
 */
export class SpeechT5ForTextToSpeech extends SpeechT5PreTrainedModel {
    /**
     * @typedef {Object} SpeechOutput
     * @property {Tensor} [spectrogram] The predicted log-mel spectrogram of shape
     * `(output_sequence_length, config.num_mel_bins)`. Returned when no `vocoder` is provided
     * @property {Tensor} [waveform] The predicted waveform of shape `(num_frames,)`. Returned when a `vocoder` is provided.
     * @property {Tensor} [cross_attentions] The outputs of the decoder's cross-attention layers of shape
     * `(config.decoder_layers, config.decoder_attention_heads, output_sequence_length, input_sequence_length)`. returned when `output_cross_attentions` is `true`.
     */

    /**
     * Converts a sequence of input tokens into a sequence of mel spectrograms, which are subsequently turned into a speech waveform using a vocoder.
     * @param {Tensor} input_values Indices of input sequence tokens in the vocabulary.
     * @param {Tensor} speaker_embeddings Tensor containing the speaker embeddings.
     * @param {Object} options Optional parameters for generating speech.
     * @param {number} [options.threshold=0.5] The generated sequence ends when the predicted stop token probability exceeds this value.
     * @param {number} [options.minlenratio=0.0] Used to calculate the minimum required length for the output sequence.
     * @param {number} [options.maxlenratio=20.0] Used to calculate the maximum allowed length for the output sequence.
     * @param {Object} [options.vocoder=null] The vocoder that converts the mel spectrogram into a speech waveform. If `null`, the output is the mel spectrogram.
     * @param {boolean} [options.output_cross_attentions=false] Whether or not to return the attentions tensors of the decoder's cross-attention layers.
     * @returns {Promise<SpeechOutput>} A promise which resolves to an object containing the spectrogram, waveform, and cross-attention tensors.
     */
    async generate_speech(
        input_values,
        speaker_embeddings,
        {
            threshold = 0.5,
            minlenratio = 0.0,
            maxlenratio = 20.0,
            vocoder = null,
            // output_cross_attentions = false, // TODO add
        } = {},
    ) {
        const model_inputs = {
            input_ids: input_values,
        };

        const { encoder_outputs, encoder_attention_mask } = await encoder_forward(this, model_inputs);

        // @ts-expect-error TS2339
        const r = encoder_outputs.dims[1] / this.config.reduction_factor;
        const maxlen = Math.floor(r * maxlenratio);
        const minlen = Math.floor(r * minlenratio);

        // @ts-expect-error TS2339
        const num_mel_bins = this.config.num_mel_bins;

        let spectrogramParts = [];
        let past_key_values = null;
        let decoder_outputs = null;
        let idx = 0;

        while (true) {
            ++idx;

            const use_cache_branch = boolTensor(!!decoder_outputs);
            let output_sequence;
            if (decoder_outputs) {
                output_sequence = decoder_outputs.output_sequence_out;
            } else {
                output_sequence = new Tensor('float32', new Float32Array(num_mel_bins), [1, 1, num_mel_bins]);
            }
            let decoderFeeds = {
                use_cache_branch,
                output_sequence,
                encoder_attention_mask: encoder_attention_mask,
                speaker_embeddings: speaker_embeddings,
                encoder_hidden_states: encoder_outputs,
            };

            this.addPastKeyValues(decoderFeeds, past_key_values);
            decoder_outputs = await sessionRun(this.sessions['decoder_model_merged'], decoderFeeds);
            past_key_values = this.getPastKeyValues(decoder_outputs, past_key_values);

            const { prob, spectrum } = decoder_outputs;
            spectrogramParts.push(spectrum);

            if (
                idx >= minlen &&
                // Finished when stop token or maximum length is reached.
                (Array.from(prob.data).filter((p) => p >= threshold).length > 0 || idx >= maxlen)
            ) {
                break;
            }
        }

        const spectrogram = cat(spectrogramParts);
        const { waveform } = await sessionRun(vocoder.sessions['model'], { spectrogram });

        return {
            spectrogram,
            waveform,
            // cross_attentions: null, // TODO add
        };
    }
}

/**
 * HiFi-GAN vocoder.
 *
 * See [SpeechT5ForSpeechToText](./models#module_models.SpeechT5ForSpeechToText) for example usage.
 */
export class SpeechT5HifiGan extends PreTrainedModel {
    main_input_name = 'spectrogram';
}
