import { PreTrainedTokenizer, ModernBertModel, ModernBertForMaskedLM, ModernBertForSequenceClassification, ModernBertForTokenClassification } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("ModernBertModel", () => {
    const model_id = "hf-internal-testing/tiny-random-ModernBertModel";

    /** @type {ModernBertModel} */
    let model;
    /** @type {PreTrainedTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await ModernBertModel.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await PreTrainedTokenizer.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "batch_size=1",
      async () => {
        const inputs = tokenizer("hello");
        const { last_hidden_state } = await model(inputs);
        expect(last_hidden_state.dims).toEqual([1, 3, 32]);
        expect(last_hidden_state.mean().item()).toBeCloseTo(-0.08922556787729263, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batch_size>1",
      async () => {
        const inputs = tokenizer(["hello", "hello world"], { padding: true });
        const { last_hidden_state } = await model(inputs);
        expect(last_hidden_state.dims).toEqual([2, 4, 32]);
        expect(last_hidden_state.mean().item()).toBeCloseTo(0.048988230526447296, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "sequence_length > local_attention_window",
      async () => {
        const text = "The sun cast long shadows across the weathered cobblestones as Thomas made his way through the ancient city. The evening air carried whispers of autumn, rustling through the golden leaves that danced and swirled around his feet. His thoughts wandered to the events that had brought him here, to this moment, in this forgotten corner of the world. The old buildings loomed above him, their facades telling stories of centuries past. Windows reflected the dying light of day, creating a kaleidoscope of amber and rose that painted the narrow streets. The distant sound of church bells echoed through the maze of alleyways, marking time's steady march forward. In his pocket, he fingered the small brass key that had belonged to his grandfather. Its weight seemed to grow heavier with each step, a tangible reminder of the promise he had made. The mystery of its purpose had consumed his thoughts for weeks, leading him through archives and dusty libraries, through conversations with local historians and elderly residents who remembered the old days. As the evening deepened into dusk, streetlamps flickered to life one by one, creating pools of warm light that guided his way. The smell of wood smoke and distant cooking fires drifted through the air, reminding him of childhood evenings spent by the hearth, listening to his grandfather's tales of hidden treasures and secret passages. His footsteps echoed against the stone walls, a rhythmic accompaniment to his journey. Each step brought him closer to his destination, though uncertainty still clouded his mind about what he might find. The old map in his other pocket, creased and worn from constant consultation, had led him this far. The street ahead narrowed, and the buildings seemed to lean in closer, their upper stories nearly touching above his head. The air grew cooler in this shadowed passage, and his breath formed small clouds in front of him. Something about this place felt different, charged with possibility and ancient secrets. He walked down the [MASK]";
        const inputs = tokenizer(text);
        const { last_hidden_state } = await model(inputs);
        expect(last_hidden_state.dims).toEqual([1, 397, 32]);
        expect(last_hidden_state.mean().item()).toBeCloseTo(-0.06889555603265762, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("ModernBertForMaskedLM", () => {
    const model_id = "hf-internal-testing/tiny-random-ModernBertForMaskedLM";

    const texts = ["The goal of life is [MASK].", "Paris is the [MASK] of France."];

    /** @type {ModernBertForMaskedLM} */
    let model;
    /** @type {PreTrainedTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await ModernBertForMaskedLM.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await PreTrainedTokenizer.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "batch_size=1",
      async () => {
        const inputs = tokenizer(texts[0]);
        const { logits } = await model(inputs);
        expect(logits.dims).toEqual([1, 9, 50368]);
        expect(logits.mean().item()).toBeCloseTo(0.0053214821964502335, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batch_size>1",
      async () => {
        const inputs = tokenizer(texts, { padding: true });
        const { logits } = await model(inputs);
        expect(logits.dims).toEqual([2, 9, 50368]);
        expect(logits.mean().item()).toBeCloseTo(0.009154772385954857, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("ModernBertForSequenceClassification", () => {
    const model_id = "hf-internal-testing/tiny-random-ModernBertForSequenceClassification";

    /** @type {ModernBertForSequenceClassification} */
    let model;
    /** @type {PreTrainedTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await ModernBertForSequenceClassification.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await PreTrainedTokenizer.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "batch_size=1",
      async () => {
        const inputs = tokenizer("hello");
        const { logits } = await model(inputs);
        const target = [[-0.7050137519836426, 2.343430519104004]];
        expect(logits.dims).toEqual([1, 2]);
        expect(logits.tolist()).toBeCloseToNested(target, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batch_size>1",
      async () => {
        const inputs = tokenizer(["hello", "hello world"], { padding: true });
        const { logits } = await model(inputs);
        const target = [
          [-0.7050137519836426, 2.343430519104004],
          [-2.6860175132751465, 3.993380546569824],
        ];
        expect(logits.dims).toEqual([2, 2]);
        expect(logits.tolist()).toBeCloseToNested(target, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("ModernBertForTokenClassification", () => {
    const model_id = "hf-internal-testing/tiny-random-ModernBertForTokenClassification";

    /** @type {ModernBertForTokenClassification} */
    let model;
    /** @type {PreTrainedTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await ModernBertForTokenClassification.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      tokenizer = await PreTrainedTokenizer.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "batch_size=1",
      async () => {
        const inputs = tokenizer("hello");
        const { logits } = await model(inputs);
        expect(logits.dims).toEqual([1, 3, 2]);
        expect(logits.mean().item()).toBeCloseTo(1.0337047576904297, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batch_size>1",
      async () => {
        const inputs = tokenizer(["hello", "hello world"], { padding: true });
        const { logits } = await model(inputs);
        expect(logits.dims).toEqual([2, 4, 2]);
        expect(logits.mean().item()).toBeCloseTo(-1.3397092819213867, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
