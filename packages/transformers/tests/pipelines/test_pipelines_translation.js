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
      await pipe?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("Translation (Marian model, target language codes)", () => {
    const model_id = "Xenova/opus-mt-en-ROMANCE";

    /** @type {TranslationPipeline} */
    let pipe;
    beforeAll(async () => {
      pipe = await pipeline(PIPELINE_ID, model_id, { dtype: "q8" });
    }, MAX_MODEL_LOAD_TIME);

    it("should be an instance of TranslationPipeline", () => {
      expect(pipe).toBeInstanceOf(TranslationPipeline);
    });

    describe("batch_size=1", () => {
      it(
        "translates into the language given by the language code",
        async () => {
          const text = "Hello, how are you?";
          expect(await pipe(`>>fr<< ${text}`)).toEqual([{ translation_text: "Bonjour, comment allez-vous?" }]);
          expect(await pipe(`>>es<< ${text}`)).toEqual([{ translation_text: "Hola, ¿cómo estás?" }]);
          expect(await pipe(`>>it<< ${text}`)).toEqual([{ translation_text: "Ciao, come stai?" }]);
          expect(await pipe(`>>pt<< ${text}`)).toEqual([{ translation_text: "Olá, como está?" }]);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    afterAll(async () => {
      await pipe?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
