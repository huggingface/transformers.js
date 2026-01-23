import { Pipeline, prepareImages } from './_base.js';

import { Tensor } from '../utils/tensor.js';

/**
 * @typedef {import('./_base.js').TextImagePipelineConstructorArgs} TextImagePipelineConstructorArgs
 * @typedef {import('./_base.js').Disposable} Disposable
 * @typedef {import('./_base.js').ImageInput} ImageInput
 */

/**
 * @typedef {Object} DocumentQuestionAnsweringSingle
 * @property {string} answer The generated text.
 * @typedef {DocumentQuestionAnsweringSingle[]} DocumentQuestionAnsweringOutput
 *
 * @callback DocumentQuestionAnsweringPipelineCallback Answer the question given as input by using the document.
 * @param {ImageInput|ImageInput[]} image The image of the document to use.
 * @param {string} question A question to ask of the document.
 * @param {Partial<import('../generation/configuration_utils.js').GenerationConfig>} [options] Additional keyword arguments to pass along to the generate method of the model.
 * @returns {Promise<DocumentQuestionAnsweringOutput>} An object (or array of objects) containing the answer(s).
 *
 * @typedef {TextImagePipelineConstructorArgs & DocumentQuestionAnsweringPipelineCallback & Disposable} DocumentQuestionAnsweringPipelineType
 */

/**
 * Document Question Answering pipeline using any `AutoModelForDocumentQuestionAnswering`.
 * The inputs/outputs are similar to the (extractive) question answering pipeline; however,
 * the pipeline takes an image (and optional OCR'd words/boxes) as input instead of text context.
 *
 * **Example:** Answer questions about a document with `Xenova/donut-base-finetuned-docvqa`.
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const qa_pipeline = await pipeline('document-question-answering', 'Xenova/donut-base-finetuned-docvqa');
 * const image = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/invoice.png';
 * const question = 'What is the invoice number?';
 * const output = await qa_pipeline(image, question);
 * // [{ answer: 'us-001' }]
 * ```
 */
export class DocumentQuestionAnsweringPipeline
    extends /** @type {new (options: TextImagePipelineConstructorArgs) => DocumentQuestionAnsweringPipelineType} */ (
        Pipeline
    )
{
    async _call(image, question, generate_kwargs = {}) {
        if (Array.isArray(image)) {
            if (image.length !== 1) {
                throw Error('Document Question Answering pipeline currently only supports a batch size of 1.');
            }
            image = image[0];
        }

        // Preprocess image
        const preparedImage = (await prepareImages(image))[0];
        const { pixel_values } = await this.processor(preparedImage);

        // Run tokenization
        const task_prompt = `<s_docvqa><s_question>${question}</s_question><s_answer>`;
        const decoder_input_ids = this.tokenizer(task_prompt, {
            add_special_tokens: false,
            padding: true,
            truncation: true,
        }).input_ids;

        // Run model
        const output = await this.model.generate({
            inputs: pixel_values,
            // @ts-expect-error Ts2339
            max_length: this.model.config.decoder.max_position_embeddings,
            decoder_input_ids,
            ...generate_kwargs,
        });

        // Decode output
        const decoded = this.tokenizer.batch_decode(/** @type {Tensor} */ (output))[0];

        // Parse answer
        const match = decoded.match(/<s_answer>(.*?)<\/s_answer>/);
        let answer = null;
        if (match && match.length >= 2) {
            answer = match[1].trim();
        }
        return [{ answer }];
    }
}
