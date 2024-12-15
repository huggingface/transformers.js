import { pipeline, TranslationPipeline } from "../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";

const PIPELINE_ID = "translation";

export default () => {
  describe("Translation", () => {
    const model_id = "Xenova/tiny-random-M2M100ForConditionalGeneration";

    /** @type {TranslationPipeline} */
    let pipe;
    beforeAll(async () => {
      pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
    }, MAX_MODEL_LOAD_TIME);

    it("should be an instance of TranslationPipeline", () => {
      expect(pipe).toBeInstanceOf(TranslationPipeline);
    });

    describe("batch_size=1", () => {
      it(
        "default",
        async () => {
          const text = "जीवन एक चॉकलेट बॉक्स की तरह है।";
          const output = await pipe(text, {
            src_lang: "hi",
            tgt_lang: "fr",
            max_new_tokens: 5,
          });
          const target = [{ translation_text: "Slovenska төсли төсли төсли" }];
          expect(output).toEqual(target);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    afterAll(async () => {
      await pipe.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
