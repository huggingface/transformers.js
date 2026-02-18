/**
 * @file Definitions of all models available in Transformers.js.
 *
 * **Example:** Load and run an `AutoModel`.
 *
 * ```javascript
 * import { AutoModel, AutoTokenizer } from '@huggingface/transformers';
 *
 * const tokenizer = await AutoTokenizer.from_pretrained('Xenova/bert-base-uncased');
 * const model = await AutoModel.from_pretrained('Xenova/bert-base-uncased');
 *
 * const inputs = await tokenizer('I love transformers!');
 * const { logits } = await model(inputs);
 * // Tensor {
 * //     data: Float32Array(183132) [-7.117443084716797, -7.107812881469727, -7.092104911804199, ...]
 * //     dims: (3) [1, 6, 30522],
 * //     type: "float32",
 * //     size: 183132,
 * // }
 * ```
 *
 * We also provide other `AutoModel`s (listed below), which you can use in the same way as the Python library. For example:
 *
 * **Example:** Load and run an `AutoModelForSeq2SeqLM`.
 * ```javascript
 * import { AutoModelForSeq2SeqLM, AutoTokenizer } from '@huggingface/transformers';
 *
 * const tokenizer = await AutoTokenizer.from_pretrained('Xenova/t5-small');
 * const model = await AutoModelForSeq2SeqLM.from_pretrained('Xenova/t5-small');
 *
 * const { input_ids } = await tokenizer('translate English to German: I love transformers!');
 * const outputs = await model.generate(input_ids);
 * const decoded = tokenizer.decode(outputs[0], { skip_special_tokens: true });
 * // 'Ich liebe Transformatoren!'
 * ```
 *
 * @module models
 */

import { AutoConfig } from '../../configs.js';
import { PreTrainedModel } from '../modeling_utils.js';

import { CUSTOM_ARCHITECTURES, MODEL_CLASS_TYPE_MAPPING, MODEL_MAPPINGS } from '../registry.js';

import * as ALL_MODEL_FILES from '../models.js';

/**
 * Base class of all AutoModels. Contains the `from_pretrained` function
 * which is used to instantiate pretrained models.
 */
class PretrainedMixin {
    /**
     * Mapping from model type to model class.
     * @type {Map<string, Object>[]}
     */
    static MODEL_CLASS_MAPPINGS = null;

    /**
     * Whether to attempt to instantiate the base class (`PretrainedModel`) if
     * the model type is not found in the mapping.
     */
    static BASE_IF_FAIL = false;

    /** @type {typeof PreTrainedModel.from_pretrained} */
    static async from_pretrained(
        pretrained_model_name_or_path,
        {
            progress_callback = null,
            config = null,
            cache_dir = null,
            local_files_only = false,
            revision = 'main',
            model_file_name = null,
            subfolder = 'onnx',
            device = null,
            dtype = null,
            use_external_data_format = null,
            session_options = {},
        } = {},
    ) {
        const options = {
            progress_callback,
            config,
            cache_dir,
            local_files_only,
            revision,
            model_file_name,
            subfolder,
            device,
            dtype,
            use_external_data_format,
            session_options,
        };
        options.config = await AutoConfig.from_pretrained(pretrained_model_name_or_path, options);

        if (!this.MODEL_CLASS_MAPPINGS) {
            throw new Error('`MODEL_CLASS_MAPPINGS` not implemented for this type of `AutoClass`: ' + this.name);
        }
        const model_type = options.config.model_type;
        for (const MODEL_CLASS_MAPPING of this.MODEL_CLASS_MAPPINGS) {
            let modelInfo = MODEL_CLASS_MAPPING.get(model_type);
            if (!modelInfo) {
                // As a fallback, we check if model_type is specified as the exact class
                for (const cls of MODEL_CLASS_MAPPING.values()) {
                    if (cls[0] === model_type) {
                        modelInfo = cls;
                        break;
                    }
                }
                if (!modelInfo) continue; // Item not found in this mapping
            }
            return await ALL_MODEL_FILES[modelInfo].from_pretrained(pretrained_model_name_or_path, options);
        }

        if (this.BASE_IF_FAIL) {
            if (!CUSTOM_ARCHITECTURES.has(model_type)) {
                console.warn(`Unknown model class "${model_type}", attempting to construct from base class.`);
            }
            return await PreTrainedModel.from_pretrained(pretrained_model_name_or_path, options);
        } else {
            throw Error(`Unsupported model type: ${model_type}`);
        }
    }
}

/**
 * Helper class which is used to instantiate pretrained models with the `from_pretrained` function.
 * The chosen model class is determined by the type specified in the model config.
 *
 * @example
 * const model = await AutoModel.from_pretrained('Xenova/bert-base-uncased');
 */
export class AutoModel extends PretrainedMixin {
    /** @type {Map<string, Object>[]} */
    // @ts-ignore
    static MODEL_CLASS_MAPPINGS = MODEL_CLASS_TYPE_MAPPING.map((x) => x[0]);
    static BASE_IF_FAIL = true;
}

/**
 * Helper class which is used to instantiate pretrained sequence classification models with the `from_pretrained` function.
 * The chosen model class is determined by the type specified in the model config.
 *
 * @example
 * const model = await AutoModelForSequenceClassification.from_pretrained('Xenova/distilbert-base-uncased-finetuned-sst-2-english');
 */
export class AutoModelForSequenceClassification extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_MAPPINGS.MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES];
}

/**
 * Helper class which is used to instantiate pretrained token classification models with the `from_pretrained` function.
 * The chosen model class is determined by the type specified in the model config.
 *
 * @example
 * const model = await AutoModelForTokenClassification.from_pretrained('Xenova/distilbert-base-multilingual-cased-ner-hrl');
 */
export class AutoModelForTokenClassification extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_MAPPINGS.MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES];
}

/**
 * Helper class which is used to instantiate pretrained sequence-to-sequence models with the `from_pretrained` function.
 * The chosen model class is determined by the type specified in the model config.
 *
 * @example
 * const model = await AutoModelForSeq2SeqLM.from_pretrained('Xenova/t5-small');
 */
export class AutoModelForSeq2SeqLM extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_MAPPINGS.MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES];
}

/**
 * Helper class which is used to instantiate pretrained sequence-to-sequence speech-to-text models with the `from_pretrained` function.
 * The chosen model class is determined by the type specified in the model config.
 *
 * @example
 * const model = await AutoModelForSpeechSeq2Seq.from_pretrained('openai/whisper-tiny.en');
 */
export class AutoModelForSpeechSeq2Seq extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_MAPPINGS.MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES];
}

/**
 * Helper class which is used to instantiate pretrained sequence-to-sequence text-to-spectrogram models with the `from_pretrained` function.
 * The chosen model class is determined by the type specified in the model config.
 *
 * @example
 * const model = await AutoModelForTextToSpectrogram.from_pretrained('microsoft/speecht5_tts');
 */
export class AutoModelForTextToSpectrogram extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_MAPPINGS.MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES];
}

/**
 * Helper class which is used to instantiate pretrained text-to-waveform models with the `from_pretrained` function.
 * The chosen model class is determined by the type specified in the model config.
 *
 * @example
 * const model = await AutoModelForTextToSpectrogram.from_pretrained('facebook/mms-tts-eng');
 */
export class AutoModelForTextToWaveform extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_MAPPINGS.MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES];
}

/**
 * Helper class which is used to instantiate pretrained causal language models with the `from_pretrained` function.
 * The chosen model class is determined by the type specified in the model config.
 *
 * @example
 * const model = await AutoModelForCausalLM.from_pretrained('Xenova/gpt2');
 */
export class AutoModelForCausalLM extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_MAPPINGS.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES];
}

/**
 * Helper class which is used to instantiate pretrained masked language models with the `from_pretrained` function.
 * The chosen model class is determined by the type specified in the model config.
 *
 * @example
 * const model = await AutoModelForMaskedLM.from_pretrained('Xenova/bert-base-uncased');
 */
export class AutoModelForMaskedLM extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_MAPPINGS.MODEL_FOR_MASKED_LM_MAPPING_NAMES];
}

/**
 * Helper class which is used to instantiate pretrained question answering models with the `from_pretrained` function.
 * The chosen model class is determined by the type specified in the model config.
 *
 * @example
 * const model = await AutoModelForQuestionAnswering.from_pretrained('Xenova/distilbert-base-cased-distilled-squad');
 */
export class AutoModelForQuestionAnswering extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_MAPPINGS.MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES];
}

/**
 * Helper class which is used to instantiate pretrained vision-to-sequence models with the `from_pretrained` function.
 * The chosen model class is determined by the type specified in the model config.
 *
 * @example
 * const model = await AutoModelForVision2Seq.from_pretrained('Xenova/vit-gpt2-image-captioning');
 */
export class AutoModelForVision2Seq extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_MAPPINGS.MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES];
}

/**
 * Helper class which is used to instantiate pretrained image classification models with the `from_pretrained` function.
 * The chosen model class is determined by the type specified in the model config.
 *
 * @example
 * const model = await AutoModelForImageClassification.from_pretrained('Xenova/vit-base-patch16-224');
 */
export class AutoModelForImageClassification extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_MAPPINGS.MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES];
}

/**
 * Helper class which is used to instantiate pretrained image segmentation models with the `from_pretrained` function.
 * The chosen model class is determined by the type specified in the model config.
 *
 * @example
 * const model = await AutoModelForImageSegmentation.from_pretrained('Xenova/detr-resnet-50-panoptic');
 */
export class AutoModelForImageSegmentation extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_MAPPINGS.MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES];
}

/**
 * Helper class which is used to instantiate pretrained image segmentation models with the `from_pretrained` function.
 * The chosen model class is determined by the type specified in the model config.
 *
 * @example
 * const model = await AutoModelForSemanticSegmentation.from_pretrained('nvidia/segformer-b3-finetuned-cityscapes-1024-1024');
 */
export class AutoModelForSemanticSegmentation extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_MAPPINGS.MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES];
}

/**
 * Helper class which is used to instantiate pretrained universal image segmentation models with the `from_pretrained` function.
 * The chosen model class is determined by the type specified in the model config.
 *
 * @example
 * const model = await AutoModelForUniversalSegmentation.from_pretrained('hf-internal-testing/tiny-random-MaskFormerForInstanceSegmentation');
 */
export class AutoModelForUniversalSegmentation extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_MAPPINGS.MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES];
}

/**
 * Helper class which is used to instantiate pretrained object detection models with the `from_pretrained` function.
 * The chosen model class is determined by the type specified in the model config.
 *
 * @example
 * const model = await AutoModelForObjectDetection.from_pretrained('Xenova/detr-resnet-50');
 */
export class AutoModelForObjectDetection extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_MAPPINGS.MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES];
}

export class AutoModelForZeroShotObjectDetection extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_MAPPINGS.MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES];
}

/**
 * Helper class which is used to instantiate pretrained mask generation models with the `from_pretrained` function.
 * The chosen model class is determined by the type specified in the model config.
 *
 * @example
 * const model = await AutoModelForMaskGeneration.from_pretrained('Xenova/sam-vit-base');
 */
export class AutoModelForMaskGeneration extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_MAPPINGS.MODEL_FOR_MASK_GENERATION_MAPPING_NAMES];
}

export class AutoModelForCTC extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_MAPPINGS.MODEL_FOR_CTC_MAPPING_NAMES];
}

export class AutoModelForAudioClassification extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_MAPPINGS.MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES];
}

export class AutoModelForXVector extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_MAPPINGS.MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES];
}

export class AutoModelForAudioFrameClassification extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_MAPPINGS.MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES];
}

export class AutoModelForDocumentQuestionAnswering extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_MAPPINGS.MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES];
}

export class AutoModelForImageMatting extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_MAPPINGS.MODEL_FOR_IMAGE_MATTING_MAPPING_NAMES];
}

export class AutoModelForImageToImage extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_MAPPINGS.MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES];
}

export class AutoModelForDepthEstimation extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_MAPPINGS.MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES];
}

export class AutoModelForNormalEstimation extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_MAPPINGS.MODEL_FOR_NORMAL_ESTIMATION_MAPPING_NAMES];
}

export class AutoModelForPoseEstimation extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_MAPPINGS.MODEL_FOR_POSE_ESTIMATION_MAPPING_NAMES];
}

export class AutoModelForImageFeatureExtraction extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_MAPPINGS.MODEL_FOR_IMAGE_FEATURE_EXTRACTION_MAPPING_NAMES];
}

export class AutoModelForImageTextToText extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_MAPPINGS.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES];
}

export class AutoModelForAudioTextToText extends PretrainedMixin {
    static MODEL_CLASS_MAPPINGS = [MODEL_MAPPINGS.MODEL_FOR_AUDIO_TEXT_TO_TEXT_MAPPING_NAMES];
}
