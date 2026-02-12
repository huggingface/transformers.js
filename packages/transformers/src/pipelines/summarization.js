import { Text2TextGenerationPipeline } from './text2text-generation.js';

/**
 * @typedef {import('./_base.js').TextPipelineConstructorArgs} TextPipelineConstructorArgs
 * @typedef {import('./_base.js').Disposable} Disposable
 */

/**
 * @typedef {Object} SummarizationSingle
 * @property {string} summary_text The summary text.
 * @typedef {SummarizationSingle[]} SummarizationOutput
 *
 * @callback SummarizationPipelineCallback Summarize the text(s) given as inputs.
 * @param {string|string[]} texts One or several articles (or one list of articles) to summarize.
 * @param {import('../generation/parameters.js').GenerationFunctionParameters} [options] Additional keyword arguments to pass along to the generate method of the model.
 * @returns {Promise<SummarizationOutput>}
 *
 * @typedef {TextPipelineConstructorArgs & SummarizationPipelineCallback & Disposable} SummarizationPipelineType
 */

/**
 * A pipeline for summarization tasks, inheriting from Text2TextGenerationPipeline.
 *
 * **Example:** Summarization w/ `Xenova/distilbart-cnn-6-6`.
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const summarizer = await pipeline('summarization', 'Xenova/distilbart-cnn-6-6');
 * const text = 'The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, ' +
 *   'and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. ' +
 *   'During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest ' +
 *   'man-made structure in the world, a title it held for 41 years until the Chrysler Building in New ' +
 *   'York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to ' +
 *   'the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the ' +
 *   'Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second ' +
 *   'tallest free-standing structure in France after the Millau Viaduct.';
 * const output = await summarizer(text, {
 *   max_new_tokens: 100,
 * });
 * // [{ summary_text: ' The Eiffel Tower is about the same height as an 81-storey building and the tallest structure in Paris. It is the second tallest free-standing structure in France after the Millau Viaduct.' }]
 * ```
 */
export class SummarizationPipeline
    extends /** @type {new (options: TextPipelineConstructorArgs) => SummarizationPipelineType} */ (
        /** @type {any} */ (Text2TextGenerationPipeline)
    )
{
    /** @type {'summary_text'} */
    _key = 'summary_text';
}
