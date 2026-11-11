import { Qwen2VLProcessor, Qwen2VLForConditionalGeneration, Qwen2VLForCausalLM, AutoTokenizer, RawImage, Tensor } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  const CONVERSATION = [
    {
      role: "user",
      content: [{ type: "text", text: "Hello" }],
    },
  ];

  // Example adapted from https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
  const CONVERSATION_WITH_IMAGE = [
    {
      role: "user",
      content: [{ type: "image" }, { type: "text", text: "Describe this image." }],
    },
  ];
  // Empty white image
  const dims = [224, 224, 3];
  const image = new RawImage(new Uint8ClampedArray(dims[0] * dims[1] * dims[2]).fill(255), ...dims);

  describe("Qwen2VLForConditionalGeneration text-only rope index", () => {
    // No weights are loaded: only the position_ids bookkeeping is exercised.
    // Reference: transformers `Qwen2VLModel.get_rope_index` (text-only branch), where
    // `mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]`.
    const createModel = () => {
      const model = Object.create(Qwen2VLForConditionalGeneration.prototype);
      model.config = { model_type: "qwen2_vl", vision_config: { spatial_merge_size: 2 } };
      model.sessions = { model: { inputNames: ["input_ids", "attention_mask", "position_ids", "past_key_values.0.key"] } };
      return model;
    };
    const int64 = (rows) => new Tensor("int64", BigInt64Array.from(rows.flat().map(BigInt)), [rows.length, rows[0].length]);
    const toNumbers = (tensor) => Array.from(tensor.data, Number);

    it("unpadded prompt: positions are arange and the delta is 0", () => {
      const model = createModel();
      const input_ids = int64([[5, 6, 7, 8]]);
      const [position_ids, rope_deltas] = model.get_rope_index(input_ids, null, null, int64([[1, 1, 1, 1]]));

      expect(position_ids.dims).toEqual([3, 1, 4]);
      expect(toNumbers(position_ids)).toEqual([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]);
      expect(rope_deltas.dims).toEqual([1, 1]);
      expect(toNumbers(rope_deltas)).toEqual([0]);
    });

    it("left padding: the delta is minus the number of padding tokens", () => {
      const model = createModel();
      const input_ids = int64([
        [0, 0, 5, 6, 7],
        [5, 6, 7, 8, 9],
      ]);
      const attention_mask = int64([
        [0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1],
      ]);
      const [position_ids, rope_deltas] = model.get_rope_index(input_ids, null, null, attention_mask);

      expect(toNumbers(position_ids).slice(0, 10)).toEqual([1, 1, 0, 1, 2, 0, 1, 2, 3, 4]);
      expect(toNumbers(rope_deltas)).toEqual([-2, 0]);
    });

    it("no attention mask: every row is arange, also for batch_size > 1", () => {
      const model = createModel();
      const [position_ids, rope_deltas] = model.get_rope_index(
        int64([
          [5, 6, 7],
          [8, 9, 10],
        ]),
        null,
        null,
        null,
      );

      expect(position_ids.dims).toEqual([3, 2, 3]);
      expect(toNumbers(position_ids)).toEqual([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]);
      expect(toNumbers(rope_deltas)).toEqual([0, 0]);
    });

    it("cached decode steps continue right after the prompt", () => {
      const model = createModel();
      const prompt_length = 16;
      let model_inputs = model.prepare_inputs_for_generation(
        null,
        {
          input_ids: int64([Array.from({ length: prompt_length }, (_, i) => i + 5)]),
          attention_mask: int64([Array(prompt_length).fill(1)]),
        },
        {},
      );
      expect(toNumbers(model_inputs.position_ids).slice(0, prompt_length)).toEqual(Array.from({ length: prompt_length }, (_, i) => i));

      const decode_positions = [];
      for (let step = 0; step < 3; ++step) {
        const past_length = prompt_length + step;
        model_inputs = model.prepare_inputs_for_generation(
          null,
          {
            ...model_inputs,
            input_ids: int64([[42]]),
            attention_mask: int64([Array(past_length + 1).fill(1)]),
            past_key_values: { get_seq_length: () => past_length },
            position_ids: null,
          },
          {},
        );
        expect(model_inputs.position_ids.dims).toEqual([3, 1, 1]);
        decode_positions.push(toNumbers(model_inputs.position_ids));
      }
      expect(decode_positions).toEqual([
        [16, 16, 16],
        [17, 17, 17],
        [18, 18, 18],
      ]);
    });

    it("cached decode step of a left-padded batch continues each row after its own tokens", () => {
      const model = createModel();
      const attention_mask = int64([
        [0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1],
      ]);
      const prefill = model.prepare_inputs_for_generation(
        null,
        {
          input_ids: int64([
            [0, 0, 5, 6, 7],
            [5, 6, 7, 8, 9],
          ]),
          attention_mask,
        },
        {},
      );
      const decode = model.prepare_inputs_for_generation(
        null,
        {
          ...prefill,
          input_ids: int64([[42], [42]]),
          attention_mask: int64([
            [0, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
          ]),
          past_key_values: { get_seq_length: () => 5 },
          position_ids: null,
        },
        {},
      );

      expect(decode.position_ids.dims).toEqual([3, 2, 1]);
      expect(toNumbers(decode.position_ids)).toEqual([3, 5, 3, 5, 3, 5]);
    });
  });

  describe("Qwen2VLForConditionalGeneration", () => {
    const model_id = "hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration";

    /** @type {Qwen2VLForConditionalGeneration} */
    let model;
    /** @type {Qwen2VLProcessor} */
    let processor;
    beforeAll(async () => {
      model = await Qwen2VLForConditionalGeneration.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      processor = await Qwen2VLProcessor.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "forward",
      async () => {
        const text = processor.apply_chat_template(CONVERSATION_WITH_IMAGE, {
          add_generation_prompt: true,
        });
        const inputs = await processor(text, image);
        const { logits } = await model(inputs);
        expect(logits.dims).toEqual([1, 89, 152064]);
        expect(logits.mean().item()).toBeCloseTo(-0.0011299321195110679, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "text-only (batch_size=1)",
      async () => {
        const text = processor.apply_chat_template(CONVERSATION, {
          add_generation_prompt: true,
        });
        const inputs = await processor(text);
        const generate_ids = await model.generate({
          ...inputs,
          max_new_tokens: 10,
        });

        const new_tokens = generate_ids.slice(null, [inputs.input_ids.dims.at(-1), null]);
        expect(new_tokens.tolist()).toEqual([[24284n, 63986n, 108860n, 84530n, 8889n, 23262n, 128276n, 64948n, 136757n, 138348n]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "text + image (batch_size=1)",
      async () => {
        const text = processor.apply_chat_template(CONVERSATION_WITH_IMAGE, {
          add_generation_prompt: true,
        });
        const inputs = await processor(text, image);
        const generate_ids = await model.generate({
          ...inputs,
          max_new_tokens: 10,
        });

        const new_tokens = generate_ids.slice(null, [inputs.input_ids.dims.at(-1), null]);
        expect(new_tokens.tolist()).toEqual([[24284n, 35302n, 60575n, 38679n, 113390n, 115118n, 137596n, 38241n, 96726n, 142301n]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("Qwen2VLForCausalLM", () => {
    const model_id = "hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration";

    /** @type {Qwen2VLForCausalLM} */
    let model;
    /** @type {AutoTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await Qwen2VLForCausalLM.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await AutoTokenizer.from_pretrained(model_id);
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
        expect(outputs.tolist()).toEqual([[14990n, 150610n, 22618n, 127483n, 72112n, 121460n, 58034n, 122233n, 18625n, 63632n]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
