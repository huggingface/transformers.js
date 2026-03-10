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
            console.log(`[${dtype}] forward default logits shape:`, logits.dims);
            console.log(`[${dtype}] forward default logits[0][0][0..3]:`, logits.data.slice(0, 4));
            expect(logits.dims[0]).toEqual(1); // batch
            expect(logits.dims[1]).toEqual(inputs.input_ids.dims[1]); // seq_len (all tokens)
          },
          MAX_TEST_EXECUTION_TIME,
        );

        it(
          "forward (num_logits_to_keep=0)",
          async () => {
            const inputs = tokenizer("hello");
            const num_logits_to_keep = new Tensor("int64", BigInt64Array.from([0n]), []);
            const { logits } = await model.forward({ ...inputs, num_logits_to_keep });
            console.log(`[${dtype}] forward nlk=0 logits shape:`, logits.dims);
            // num_logits_to_keep=0 should return all logits (same as default)
            expect(logits.dims[1]).toEqual(inputs.input_ids.dims[1]);
          },
          MAX_TEST_EXECUTION_TIME,
        );

        it(
          "forward (num_logits_to_keep=1)",
          async () => {
            const inputs = tokenizer("hello");
            const num_logits_to_keep = new Tensor("int64", BigInt64Array.from([1n]), []);
            const { logits } = await model.forward({ ...inputs, num_logits_to_keep });
            console.log(`[${dtype}] forward nlk=1 logits shape:`, logits.dims);
            expect(logits.dims[1]).toEqual(1); // only last token's logits
          },
          MAX_TEST_EXECUTION_TIME,
        );

        it(
          "forward (num_logits_to_keep=3)",
          async () => {
            const inputs = tokenizer("hello world");
            const num_logits_to_keep = new Tensor("int64", BigInt64Array.from([3n]), []);
            const { logits } = await model.forward({ ...inputs, num_logits_to_keep });
            console.log(`[${dtype}] forward nlk=3 logits shape:`, logits.dims);
            expect(logits.dims[1]).toEqual(3); // last 3 tokens' logits
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
            const result = outputs.tolist();
            console.log(`[${dtype}] generate result:`, JSON.stringify(result.map((r) => r.map((v) => v.toString() + "n"))));
            expect(result.length).toBeGreaterThan(0);
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
        console.log("pipeline result:", JSON.stringify(result));
        expect(result).toBeDefined();
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
