import { PreTrainedTokenizer, DeepseekV4ForCausalLM } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("DeepseekV4ForCausalLM", () => {
    const model_id = "onnx-internal-testing/tiny-random-DeepseekV4ForCausalLM";
    /** @type {DeepseekV4ForCausalLM} */
    let model;
    /** @type {PreTrainedTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await DeepseekV4ForCausalLM.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await PreTrainedTokenizer.from_pretrained(model_id);
      tokenizer.padding_side = "left";
    }, MAX_MODEL_LOAD_TIME);

    it(
      "batch_size=1",
      async () => {
        const inputs = tokenizer("hello");
        const outputs = await model.generate({
          ...inputs,
          max_length: 10,
          do_sample: false,
        });
        expect(outputs.tolist()).toEqual([[33310n, 83919n, 100040n, 101230n, 55545n, 23803n, 76294n, 119693n, 74059n, 1036n]]);
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
          do_sample: false,
        });
        expect(outputs.tolist()).toEqual([
          [1n, 33310n, 83919n, 100040n, 101230n, 55545n, 23803n, 76294n, 119693n, 74059n],
          [33310n, 2058n, 75023n, 89073n, 120077n, 49327n, 57703n, 107564n, 3563n, 126139n],
        ]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "generate w/ past_compressor and past_indexer cache",
      async () => {
        const inputs = tokenizer("hello");
        const { past_key_values } = await model.generate({
          ...inputs,
          max_new_tokens: 1,
          do_sample: false,
          return_dict_in_generate: true,
        });

        expect(past_key_values.get_seq_length()).toBeGreaterThan(0);
        const cache_names = ["past_key_values.0.key", "past_key_values.0.value", "past_key_values.3.key", "past_key_values.3.value", "past_compressor.2.kv", "past_compressor.2.gate", "past_indexer.2.kv", "past_indexer.2.gate", "past_compressor.3.kv", "past_compressor.3.gate"];
        for (const name of cache_names) {
          expect(past_key_values[name]).toBeDefined();
        }

        await past_key_values.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
