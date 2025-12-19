import { PreTrainedModel } from '../pre-trained-model.js';
import { sessionRun } from '../session.js';
import { MimiEncoderOutput, MimiDecoderOutput } from '../output.js';

export class MimiPreTrainedModel extends PreTrainedModel {
    main_input_name = 'input_values';
    forward_params = ['input_values'];
}

/**
 * The Mimi neural audio codec model.
 */
export class MimiModel extends MimiPreTrainedModel {
    /**
     * Encodes the input audio waveform into discrete codes.
     * @param {Object} inputs Model inputs
     * @param {Tensor} [inputs.input_values] Float values of the input audio waveform, of shape `(batch_size, channels, sequence_length)`).
     * @returns {Promise<MimiEncoderOutput>} The output tensor of shape `(batch_size, num_codebooks, sequence_length)`.
     */
    async encode(inputs) {
        return new MimiEncoderOutput(await sessionRun(this.sessions['encoder_model'], inputs));
    }

    /**
     * Decodes the given frames into an output audio waveform.
     * @param {MimiEncoderOutput} inputs The encoded audio codes.
     * @returns {Promise<MimiDecoderOutput>} The output tensor of shape `(batch_size, num_channels, sequence_length)`.
     */
    async decode(inputs) {
        return new MimiDecoderOutput(await sessionRun(this.sessions['decoder_model'], inputs));
    }
}

export class MimiEncoderModel extends MimiPreTrainedModel {
    /** @type {typeof PreTrainedModel.from_pretrained} */
    static async from_pretrained(pretrained_model_name_or_path, options = {}) {
        return super.from_pretrained(pretrained_model_name_or_path, {
            ...options,
            // Update default model file name if not provided
            model_file_name: options.model_file_name ?? 'encoder_model',
        });
    }
}
export class MimiDecoderModel extends MimiPreTrainedModel {
    /** @type {typeof PreTrainedModel.from_pretrained} */
    static async from_pretrained(pretrained_model_name_or_path, options = {}) {
        return super.from_pretrained(pretrained_model_name_or_path, {
            ...options,
            // Update default model file name if not provided
            model_file_name: options.model_file_name ?? 'decoder_model',
        });
    }
}
