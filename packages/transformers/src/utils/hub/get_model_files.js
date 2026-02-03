import { apis } from '../../env.js';
import {
    DATA_TYPES,
    DEFAULT_DEVICE_DTYPE_MAPPING,
    DEFAULT_DTYPE_SUFFIX_MAPPING,
    DEFAULT_DEVICE_DTYPE,
} from '../dtypes.js';
import { MODEL_TYPES, MODEL_TYPE_MAPPING } from '../../models/modeling_utils.js';
import { AutoConfig } from '../../configs.js';

/**
 * Returns the list of files that will be loaded for a model based on its configuration.
 *
 * This function reads configuration from the model's config.json on the hub.
 * If dtype/device are not specified in the config, you can provide them to match
 * what the pipeline will actually use.
 *
 * @param {string} modelId The model id (e.g., "Xenova/llama-2-7b")
 * @param {Object} [options] Optional parameters
 * @param {import('../../configs.js').PretrainedConfig} [options.config=null] Pre-loaded model config (optional, will be fetched if not provided)
 * @param {import('../dtypes.js').DataType|Record<string, import('../dtypes.js').DataType>} [options.dtype=null] Override dtype (use this if passing dtype to pipeline)
 * @param {import('../devices.js').DeviceType|Record<string, import('../devices.js').DeviceType>} [options.device=null] Override device (use this if passing device to pipeline)
 * @returns {Promise<string[]>} Array of file paths that will be loaded
 */
export async function get_model_files(
    modelId,
    { config = null, dtype: overrideDtype = null, device: overrideDevice = null } = {},
) {
    if (!config) {
        config = await AutoConfig.from_pretrained(modelId);
    }

    const files = [];
    const custom_config = config['transformers.js_config'] ?? {};

    const use_external_data_format = custom_config.use_external_data_format;
    const subfolder = 'onnx'; // Always 'onnx' as per the default in from_pretrained

    let device = overrideDevice ?? custom_config.device;
    let dtype = overrideDtype ?? custom_config.dtype;

    // Infer model type from config
    // We use MODEL_TYPE_MAPPING (the authoritative source used by from_pretrained) when possible,
    // and fall back to heuristic detection for unknown architectures
    let modelType;

    // @ts-ignore - architectures is set via Object.assign in PretrainedConfig constructor
    const architectures = /** @type {string[]} */ (config.architectures || []);

    // Try to find a known architecture in MODEL_TYPE_MAPPING
    // This ensures we use the same logic as from_pretrained()
    let foundInMapping = false;
    for (const arch of architectures) {
        const mappedType = MODEL_TYPE_MAPPING.get(arch);
        if (mappedType !== undefined) {
            modelType = mappedType;
            foundInMapping = true;
            break;
        }
    }

    // Fall back to manual detection if not found in mapping
    if (!foundInMapping) {
        if (config.is_encoder_decoder) {
            const modelName = config.model_type;
            if (['whisper', 'vision-encoder-decoder'].includes(modelName)) {
                modelType = MODEL_TYPES.Vision2Seq;
            } else if (modelName === 'musicgen') {
                modelType = MODEL_TYPES.Musicgen;
            } else {
                modelType = MODEL_TYPES.Seq2Seq;
            }
        } else {
            if (architectures.some((arch) => arch.includes('CausalLM') || arch.includes('LMHead'))) {
                modelType = MODEL_TYPES.DecoderOnly;
            } else {
                modelType = MODEL_TYPES.EncoderOnly;
            }
        }
    }

    // Helper function to determine dtype for a given file
    // This reads from config only, matching the actual loading behavior
    const get_dtype = (fileName) => {
        if (dtype && typeof dtype === 'object') {
            const fileDtype = dtype[fileName];
            if (fileDtype && fileDtype !== DATA_TYPES.auto && DATA_TYPES.hasOwnProperty(fileDtype)) {
                return fileDtype;
            }
        }

        if (dtype && typeof dtype === 'string' && dtype !== DATA_TYPES.auto && DATA_TYPES.hasOwnProperty(dtype)) {
            return dtype;
        }

        const selectedDevice = /** @type {string} */ (device ?? (apis.IS_NODE_ENV ? 'cpu' : 'wasm'));
        return DEFAULT_DEVICE_DTYPE_MAPPING[selectedDevice] ?? DEFAULT_DEVICE_DTYPE;
    };

    const add_model_file = (fileName, baseName = null) => {
        baseName = baseName ?? fileName;
        const selectedDtype = get_dtype(fileName);

        const suffix = DEFAULT_DTYPE_SUFFIX_MAPPING[selectedDtype] ?? '';
        const fullName = `${baseName}${suffix}.onnx`;
        const fullPath = subfolder ? `${subfolder}/${fullName}` : fullName;
        files.push(fullPath);

        // Check for external data files
        let external_data_format = use_external_data_format;
        if (typeof use_external_data_format === 'object' && use_external_data_format !== null) {
            external_data_format = use_external_data_format[fullName] ?? use_external_data_format[fileName] ?? false;
        }

        const num_chunks = +external_data_format; // (false=0, true=1, number remains the same)
        for (let i = 0; i < num_chunks; ++i) {
            const dataFileName = `${fullName}_data${i === 0 ? '' : '_' + i}`;
            const dataFilePath = subfolder ? `${subfolder}/${dataFileName}` : dataFileName;
            files.push(dataFilePath);
        }
    };

    // Add model files based on model type
    if (modelType === MODEL_TYPES.DecoderOnly) {
        add_model_file('model', 'model');
        files.push('generation_config.json');
    } else if (modelType === MODEL_TYPES.Seq2Seq || modelType === MODEL_TYPES.Vision2Seq) {
        add_model_file('model', 'encoder_model');
        add_model_file('decoder_model_merged');
        // Note: generation_config.json is only loaded for generation models (e.g., T5ForConditionalGeneration)
        // not for base models (e.g., T5Model). Since we can't determine the specific class here,
        // we include it as it's loaded for most use cases.
        files.push('generation_config.json');
    } else if (modelType === MODEL_TYPES.MaskGeneration) {
        add_model_file('model', 'vision_encoder');
        add_model_file('prompt_encoder_mask_decoder');
    } else if (modelType === MODEL_TYPES.EncoderDecoder) {
        add_model_file('model', 'encoder_model');
        add_model_file('decoder_model_merged');
    } else if (modelType === MODEL_TYPES.ImageTextToText) {
        add_model_file('embed_tokens');
        add_model_file('vision_encoder');
        add_model_file('decoder_model_merged');
        if (config.is_encoder_decoder) {
            add_model_file('model', 'encoder_model');
        }
        files.push('generation_config.json');
    } else if (modelType === MODEL_TYPES.AudioTextToText) {
        add_model_file('embed_tokens');
        add_model_file('audio_encoder');
        add_model_file('decoder_model_merged');
        files.push('generation_config.json');
    } else if (modelType === MODEL_TYPES.ImageAudioTextToText) {
        add_model_file('embed_tokens');
        add_model_file('audio_encoder');
        add_model_file('vision_encoder');
        add_model_file('decoder_model_merged');
        files.push('generation_config.json');
    } else if (modelType === MODEL_TYPES.Musicgen) {
        add_model_file('model', 'text_encoder');
        add_model_file('decoder_model_merged');
        add_model_file('encodec_decode');
        files.push('generation_config.json');
    } else if (modelType === MODEL_TYPES.MultiModality) {
        add_model_file('prepare_inputs_embeds');
        add_model_file('model', 'language_model');
        add_model_file('lm_head');
        add_model_file('gen_head');
        add_model_file('gen_img_embeds');
        add_model_file('image_decode');
        files.push('generation_config.json');
    } else if (modelType === MODEL_TYPES.Phi3V) {
        add_model_file('prepare_inputs_embeds');
        add_model_file('model');
        add_model_file('vision_encoder');
        files.push('generation_config.json');
    } else if (modelType === MODEL_TYPES.Chatterbox) {
        add_model_file('embed_tokens');
        add_model_file('speech_encoder');
        add_model_file('model', 'language_model');
        add_model_file('conditional_decoder');
        files.push('generation_config.json');
    } else if (modelType === MODEL_TYPES.AutoEncoder) {
        add_model_file('encoder_model');
        add_model_file('decoder_model');
    } else if (modelType === MODEL_TYPES.Supertonic) {
        add_model_file('text_encoder');
        add_model_file('latent_denoiser');
        add_model_file('voice_decoder');
    } else {
        // MODEL_TYPES.EncoderOnly or unknown
        add_model_file('model', 'model');
    }

    // Add config.json (always loaded)
    files.push('config.json');

    return files;
}
