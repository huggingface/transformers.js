/**
 * @file Processors turn raw inputs (images, audio, text) into the tensor
 * shapes a model expects. Pipelines pick the right processor automatically;
 * call one directly only when you need to preprocess without running
 * inference.
 *
 * Three `Auto*` entry points cover the common cases:
 * - `AutoProcessor` — multi-modal (tokenizer + image/audio), e.g. Whisper, CLIP.
 * - `AutoImageProcessor` — vision-only models.
 * - `AutoFeatureExtractor` — audio-only models.
 *
 * **Example:** Prepare audio for Whisper.
 * ```javascript
 * import { AutoProcessor, load_audio } from '@huggingface/transformers';
 *
 * const processor = await AutoProcessor.from_pretrained('onnx-community/whisper-tiny.en');
 * const audio = await load_audio('https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac', 16000);
 * const { input_features } = await processor(audio);
 * // Tensor {
 * //   data: Float32Array(240000) [0.4752984642982483, 0.5597258806228638, 0.56434166431427, ...],
 * //   dims: [1, 80, 3000],
 * //   type: 'float32',
 * //   size: 240000,
 * // }
 * ```
 *
 * @module processors
 */
import { PROCESSOR_NAME, CHAT_TEMPLATE_NAME } from './utils/constants.js';
import { Callable } from './utils/generic.js';
import { getModelJSON, getModelText } from './utils/hub.js';

/**
 * @typedef {Object} ProcessorProperties Additional processor-specific properties.
 * @typedef {import('./utils/hub.js').PretrainedOptions & ProcessorProperties} PretrainedProcessorOptions
 * @typedef {import('./tokenization_utils.js').PreTrainedTokenizer} PreTrainedTokenizer
 */

/**
 * Multi-modal preprocessor that delegates to the tokenizer, image processor,
 * and/or feature extractor required by a model.
 */
export class Processor extends Callable {
    static classes = ['image_processor_class', 'tokenizer_class', 'feature_extractor_class'];
    static uses_processor_config = false;
    static uses_chat_template_file = false;

    /**
     * Create a processor from parsed config and its component preprocessors.
     * @param {Object} config Processor configuration.
     * @param {Record<string, Object>} components Loaded tokenizer, image processor, and/or feature extractor.
     * @param {string|null} chat_template Optional chat template loaded from the model repo.
     */
    constructor(config, components, chat_template) {
        super();
        this.config = config;
        this.components = components;
        this.chat_template = chat_template;
    }

    /**
     * @returns {import('./image_processors_utils.js').ImageProcessor|undefined} The image processor of the processor, if it exists.
     */
    get image_processor() {
        return this.components.image_processor;
    }

    /**
     * @returns {PreTrainedTokenizer|undefined} The tokenizer of the processor, if it exists.
     */
    get tokenizer() {
        return this.components.tokenizer;
    }

    /**
     * @returns {import('./feature_extraction_utils.js').FeatureExtractor|undefined} The feature extractor of the processor, if it exists.
     */
    get feature_extractor() {
        return this.components.feature_extractor;
    }

    /**
     * Delegates to the underlying tokenizer's `apply_chat_template`.
     * @param {Parameters<PreTrainedTokenizer['apply_chat_template']>[0]} messages
     * @param {Parameters<PreTrainedTokenizer['apply_chat_template']>[1]} options
     * @returns {ReturnType<PreTrainedTokenizer['apply_chat_template']>}
     */
    apply_chat_template(messages, options = {}) {
        if (!this.tokenizer) {
            throw new Error('Unable to apply chat template without a tokenizer.');
        }
        return this.tokenizer.apply_chat_template(messages, {
            tokenize: false, // default to false
            chat_template: this.chat_template ?? undefined,
            ...options,
        });
    }

    /**
     * Decode a batch of tokenized sequences via the underlying tokenizer.
     * @param {Parameters<PreTrainedTokenizer['batch_decode']>} args
     * @returns {ReturnType<PreTrainedTokenizer['batch_decode']>}
     */
    batch_decode(...args) {
        if (!this.tokenizer) {
            throw new Error('Unable to decode without a tokenizer.');
        }
        return this.tokenizer.batch_decode(...args);
    }

    /**
     * Decode a single tokenized sequence via the underlying tokenizer.
     * @param {Parameters<PreTrainedTokenizer['decode']>} args
     * @returns {ReturnType<PreTrainedTokenizer['decode']>}
     */
    decode(...args) {
        if (!this.tokenizer) {
            throw new Error('Unable to decode without a tokenizer.');
        }
        return this.tokenizer.decode(...args);
    }

    /**
     * Calls the feature_extractor function with the given input.
     * @param {any} input The input to extract features from.
     * @param {...any} args Additional arguments.
     * @returns {Promise<any>} A Promise that resolves with the extracted features.
     */
    async _call(input, ...args) {
        for (const item of [this.image_processor, this.feature_extractor, this.tokenizer]) {
            if (item) {
                return item(input, ...args);
            }
        }
        throw new Error('No image processor, feature extractor, or tokenizer found.');
    }

    /**
     * Instantiate one of the processor classes of the library from a pretrained model.
     *
     * The processor class to instantiate is selected based on the `image_processor_type` (or `feature_extractor_type`; legacy)
     * property of the config object (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible)
     *
     * @param {string} pretrained_model_name_or_path The name or path of the pretrained model. Can be either:
     * - A string, the *model id* of a pretrained processor hosted inside a model repo on huggingface.co.
     *   Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
     *   user or organization name, like `dbmdz/bert-base-german-cased`.
     * - A path to a *directory* containing processor files, e.g., `./my_model_directory/`.
     * @param {PretrainedProcessorOptions} options Additional options for loading the processor.
     *
     * @returns {Promise<Processor>} A new processor instance.
     */
    static async from_pretrained(pretrained_model_name_or_path, options = {}) {
        const [config, components, chat_template] = await Promise.all([
            // TODO:
            this.uses_processor_config
                ? getModelJSON(pretrained_model_name_or_path, PROCESSOR_NAME, true, options)
                : {},
            Promise.all(
                this.classes
                    .filter((cls) => cls in this)
                    .map(async (cls) => {
                        const component = await this[cls].from_pretrained(pretrained_model_name_or_path, options);
                        return [cls.replace(/_class$/, ''), component];
                    }),
            ).then(Object.fromEntries),
            this.uses_chat_template_file
                ? getModelText(pretrained_model_name_or_path, CHAT_TEMPLATE_NAME, true, options)
                : null,
        ]);

        return new this(config, components, chat_template);
    }
}
