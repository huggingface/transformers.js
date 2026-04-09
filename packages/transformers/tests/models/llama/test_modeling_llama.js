import { LlamaTokenizer, LlamaForCausalLM, pipeline, Tensor } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, MAX_TEST_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("LlamaForCausalLM", () => {
    const model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM";
    /** @type {LlamaForCausalLM} */
    let model;
    /** @type {LlamaTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await LlamaForCausalLM.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await LlamaTokenizer.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "batch_size=1",
      async () => {
        const inputs = tokenizer("hello");
        const outputs = await model.generate({
          ...inputs,
          max_length: 10,
        });
        expect(outputs.tolist()).toEqual([[1n, 22172n, 18547n, 8143n, 22202n, 9456n, 17213n, 15330n, 26591n, 15721n]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batch_size>1",
      async () => {
        const inputs = tokenizer(["hello", "hello world"], { padding: true });
        const outputs = await model.generate({
          ...inputs,
          max_length: 10,
        });
        expect(outputs.tolist()).toEqual([
          [0n, 1n, 22172n, 18547n, 8143n, 22202n, 9456n, 17213n, 15330n, 26591n],
          [1n, 22172n, 3186n, 24786n, 19169n, 20222n, 29993n, 27146n, 27426n, 24562n],
        ]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("LlamaForCausalLM (onnxruntime-genai) w/ num_logits_to_keep", () => {
    const model_id = "onnx-internal-testing/tiny-random-LlamaForCausalLM_num_logits_to_keep";

    for (const dtype of ["fp32", "fp16"]) {
      describe(`dtype=${dtype}`, () => {
        /** @type {LlamaForCausalLM} */
        let model;
        /** @type {LlamaTokenizer} */
        let tokenizer;
        beforeAll(async () => {
          model = await LlamaForCausalLM.from_pretrained(model_id, {
            ...DEFAULT_MODEL_OPTIONS,
            dtype,
          });
          tokenizer = await LlamaTokenizer.from_pretrained(model_id);
        }, MAX_MODEL_LOAD_TIME);

        it(
          "forward (default num_logits_to_keep)",
          async () => {
            const inputs = tokenizer("hello");
            const { logits } = await model.forward(inputs);
            // Default: returns logits for all tokens
            expect(logits.dims).toEqual([1, 2, 128256]);
          },
          MAX_TEST_EXECUTION_TIME,
        );

        it(
          "forward (num_logits_to_keep=0)",
          async () => {
            const inputs = tokenizer("hello");
            const num_logits_to_keep = new Tensor("int64", BigInt64Array.from([0n]), []);
            const { logits } = await model.forward({ ...inputs, num_logits_to_keep });
            // num_logits_to_keep=0 returns all logits (same as default)
            expect(logits.dims).toEqual([1, 2, 128256]);
          },
          MAX_TEST_EXECUTION_TIME,
        );

        it(
          "forward (num_logits_to_keep=1)",
          async () => {
            const inputs = tokenizer("hello");
            const num_logits_to_keep = new Tensor("int64", BigInt64Array.from([1n]), []);
            const { logits } = await model.forward({ ...inputs, num_logits_to_keep });
            // Only last token's logits
            expect(logits.dims).toEqual([1, 1, 128256]);
          },
          MAX_TEST_EXECUTION_TIME,
        );

        it(
          "forward (num_logits_to_keep=3)",
          async () => {
            const inputs = tokenizer("hello world");
            const num_logits_to_keep = new Tensor("int64", BigInt64Array.from([3n]), []);
            const { logits } = await model.forward({ ...inputs, num_logits_to_keep });
            // Last 3 tokens' logits
            expect(logits.dims).toEqual([1, 3, 128256]);
          },
          MAX_TEST_EXECUTION_TIME,
        );

        it(
          "generate",
          async () => {
            const inputs = tokenizer("hello");
            const outputs = await model.generate({
              ...inputs,
              max_length: 5,
            });
            expect(outputs.tolist()).toEqual([[128000n, 15339n, 73474n, 38648n, 56375n]]);
          },
          MAX_TEST_EXECUTION_TIME,
        );

        afterAll(async () => {
          await model?.dispose();
        }, MAX_MODEL_DISPOSE_TIME);
      });
    }

    it(
      "pipeline",
      async () => {
        const pipe = await pipeline("text-generation", model_id, DEFAULT_MODEL_OPTIONS);
        const result = await pipe("hello", { max_new_tokens: 3 });
        expect(result).toEqual([{ generated_text: "hello-close heroic actively" }]);
        await pipe.dispose();
      },
      MAX_TEST_TIME,
    );
  });

  describe("LlamaForCausalLM (onnxruntime-genai)", () => {
    const model_id = "onnx-community/tiny-random-LlamaForCausalLM-ONNX";
    /** @type {LlamaTokenizer} */
    let tokenizer;
    let inputs;
    beforeAll(async () => {
      tokenizer = await LlamaTokenizer.from_pretrained(model_id);
      inputs = tokenizer("hello");
    }, MAX_MODEL_LOAD_TIME);

    const dtypes = ["fp32", "fp16", "q4", "q4f16"];

    for (const dtype of dtypes) {
      it(
        `dtype=${dtype}`,
        async () => {
          /** @type {LlamaForCausalLM} */
          const model = await LlamaForCausalLM.from_pretrained(model_id, {
            ...DEFAULT_MODEL_OPTIONS,
            dtype,
          });

          const outputs = await model.generate({
            ...inputs,
            max_length: 5,
          });
          expect(outputs.tolist()).toEqual([[128000n, 15339n, 15339n, 15339n, 15339n]]);

          await model?.dispose();
        },
        MAX_TEST_TIME,
      );
    }
  });
};
