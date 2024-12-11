import { T5Tokenizer, MusicgenForConditionalGeneration, full } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("MusicgenForConditionalGeneration", () => {
    const model_id = "hf-internal-testing/tiny-random-MusicgenForConditionalGeneration";

    // Example adapted from https://huggingface.co/docs/transformers/model_doc/musicgen#text-conditional-generation
    const texts = ["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"];

    /** @type {MusicgenForConditionalGeneration} */
    let model;
    /** @type {T5Tokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await MusicgenForConditionalGeneration.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await T5Tokenizer.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "forward",
      async () => {
        // Example from https://huggingface.co/docs/transformers/model_doc/musicgen#transformers.MusicgenForConditionalGeneration.forward.example
        const inputs = tokenizer(texts, { padding: true });
        const pad_token_id = BigInt(model.generation_config.pad_token_id);
        const decoder_input_ids = full([inputs.input_ids.dims[0] * model.config.decoder.num_codebooks, 1], pad_token_id);
        const { logits } = await model({ ...inputs, decoder_input_ids });
        expect(logits.dims).toEqual([8, 1, 99]);
        expect(logits.mean().item()).toBeCloseTo(-0.0018370470497757196, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batch_size=1",
      async () => {
        const inputs = tokenizer(texts[0]);
        const audio_values = await model.generate({ ...inputs, max_length: 10 });
        expect(audio_values.dims).toEqual([1, 1, 1920]);
        expect(audio_values.mean().item()).toBeCloseTo(0.16644205152988434, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batch_size>1",
      async () => {
        const inputs = tokenizer(texts, { padding: true });
        const audio_values = await model.generate({ ...inputs, max_length: 10 });
        expect(audio_values.dims).toEqual([2, 1, 1920]);
        expect(audio_values.mean().item()).toBeCloseTo(0.16644206643104553, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
