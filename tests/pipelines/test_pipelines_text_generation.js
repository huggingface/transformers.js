import { pipeline, TextGenerationPipeline } from "../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";

const PIPELINE_ID = "text-generation";

export default () => {
  describe("Text Generation", () => {
    const model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM";

    /** @type {TextGenerationPipeline} */
    let pipe;
    beforeAll(async () => {
      pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
    }, MAX_MODEL_LOAD_TIME);

    it("should be an instance of TextGenerationPipeline", () => {
      expect(pipe).toBeInstanceOf(TextGenerationPipeline);
    });

    describe("batch_size=1", () => {
      const text_input = "hello";
      const generated_text_target = "erdingsAndroid Load";
      const text_target = [{ generated_text: text_input + generated_text_target }];
      const new_text_target = [{ generated_text: generated_text_target }];

      const chat_input = [
        { role: "system", content: "a" },
        { role: "user", content: "b" },
      ];
      const chat_target = [
        {
          generated_text: [
            { role: "system", content: "a" },
            { role: "user", content: "b" },
            { role: "assistant", content: " Southern abund Load" },
          ],
        },
      ];

      it(
        "text input (single)",
        async () => {
          const output = await pipe(text_input, { max_new_tokens: 3 });
          expect(output).toEqual(text_target);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "text input (list)",
        async () => {
          const output = await pipe([text_input], { max_new_tokens: 3 });
          expect(output).toEqual([text_target]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "text input (single) - return_full_text=false",
        async () => {
          const output = await pipe(text_input, { max_new_tokens: 3, return_full_text: false });
          expect(output).toEqual(new_text_target);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "text input (list) - return_full_text=false",
        async () => {
          const output = await pipe([text_input], { max_new_tokens: 3, return_full_text: false });
          expect(output).toEqual([new_text_target]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "chat input (single)",
        async () => {
          const output = await pipe(chat_input, { max_new_tokens: 3 });
          expect(output).toEqual(chat_target);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "chat input (list)",
        async () => {
          const output = await pipe([chat_input], { max_new_tokens: 3 });
          expect(output).toEqual([chat_target]);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    // TODO: Fix batch_size>1
    // describe('batch_size>1', () => {
    //     it('default', async () => {
    //         const output = await pipe(['hello', 'hello world']);
    //         const target = [
    //            [{generated_text: 'helloerdingsAndroid Load'}],
    //            [{generated_text: 'hello world zerosMillнал'}],
    //         ];
    //         expect(output).toEqual(target);
    //     }, MAX_TEST_EXECUTION_TIME);
    // });

    afterAll(async () => {
      await pipe.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
