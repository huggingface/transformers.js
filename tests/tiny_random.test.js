import {
  // Pipelines
  pipeline,
  FillMaskPipeline,
  TextClassificationPipeline,
  TextGenerationPipeline,
  TranslationPipeline,
  ImageClassificationPipeline,
  ZeroShotImageClassificationPipeline,
  TokenClassificationPipeline,
  QuestionAnsweringPipeline,
  DocumentQuestionAnsweringPipeline,

  // Other
  RawImage,
} from "../src/transformers.js";

import { init, MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "./init.js";
import { compare } from "./test_utils.js";

init();

describe("Tiny random pipelines", () => {
  describe("fill-mask", () => {
    const model_id = "hf-internal-testing/tiny-random-BertForMaskedLM";

    /** @type {FillMaskPipeline} */
    let pipe;
    beforeAll(async () => {
      pipe = await pipeline("fill-mask", model_id, DEFAULT_MODEL_OPTIONS);
    }, MAX_MODEL_LOAD_TIME);

    describe("batch_size=1", () => {
      it(
        "default (top_k=5)",
        async () => {
          const output = await pipe("a [MASK] c");
          const target = [
            { score: 0.0013377574505284429, token: 854, token_str: "##ο", sequence: "aο c" },
            { score: 0.001248967950232327, token: 962, token_str: "##ち", sequence: "aち c" },
            { score: 0.0012304208939895034, token: 933, token_str: "##ع", sequence: "aع c" },
            { score: 0.0012301815440878272, token: 313, token_str: "ფ", sequence: "a ფ c" },
            { score: 0.001222139224410057, token: 624, token_str: "未", sequence: "a 未 c" },
          ];
          compare(output, target, 1e-5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "custom (top_k=2)",
        async () => {
          const output = await pipe("a [MASK] c", { top_k: 2 });
          const target = [
            { score: 0.0013377574505284429, token: 854, token_str: "##ο", sequence: "aο c" },
            { score: 0.001248967950232327, token: 962, token_str: "##ち", sequence: "aち c" },
          ];
          compare(output, target, 1e-5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    describe("batch_size>1", () => {
      it(
        "default (top_k=5)",
        async () => {
          const output = await pipe(["a [MASK] c", "a b [MASK] c"]);
          const target = [
            [
              { score: 0.0013377574505284429, token: 854, token_str: "##ο", sequence: "aο c" },
              { score: 0.001248967950232327, token: 962, token_str: "##ち", sequence: "aち c" },
              { score: 0.0012304208939895034, token: 933, token_str: "##ع", sequence: "aع c" },
              { score: 0.0012301815440878272, token: 313, token_str: "ფ", sequence: "a ფ c" },
              { score: 0.001222139224410057, token: 624, token_str: "未", sequence: "a 未 c" },
            ],
            [
              { score: 0.0013287801994010806, token: 962, token_str: "##ち", sequence: "a bち c" },
              { score: 0.0012486606137827039, token: 823, token_str: "##ن", sequence: "a bن c" },
              { score: 0.0012320734094828367, token: 1032, token_str: "##ც", sequence: "a bც c" },
              { score: 0.0012295148335397243, token: 854, token_str: "##ο", sequence: "a bο c" },
              { score: 0.0012277684872969985, token: 624, token_str: "未", sequence: "a b 未 c" },
            ],
          ];
          compare(output, target, 1e-5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "custom (top_k=2)",
        async () => {
          const output = await pipe(["a [MASK] c", "a b [MASK] c"], { top_k: 2 });
          const target = [
            [
              { score: 0.0013377574505284429, token: 854, token_str: "##ο", sequence: "aο c" },
              { score: 0.001248967950232327, token: 962, token_str: "##ち", sequence: "aち c" },
            ],
            [
              { score: 0.0013287801994010806, token: 962, token_str: "##ち", sequence: "a bち c" },
              { score: 0.0012486606137827039, token: 823, token_str: "##ن", sequence: "a bن c" },
            ],
          ];
          compare(output, target, 1e-5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    afterAll(async () => {
      await pipe?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("text-classification", () => {
    const model_id = "hf-internal-testing/tiny-random-BertForSequenceClassification";

    /** @type {TextClassificationPipeline} */
    let pipe;
    beforeAll(async () => {
      pipe = await pipeline("text-classification", model_id, DEFAULT_MODEL_OPTIONS);
    }, MAX_MODEL_LOAD_TIME);

    describe("batch_size=1", () => {
      it(
        "default (top_k=1)",
        async () => {
          const output = await pipe("a");
          const target = [{ label: "LABEL_0", score: 0.5076976418495178 }];
          compare(output, target, 1e-5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "custom (top_k=2)",
        async () => {
          const output = await pipe("a", { top_k: 2 });
          const target = [
            { label: "LABEL_0", score: 0.5076976418495178 },
            { label: "LABEL_1", score: 0.49230238795280457 },
          ];
          compare(output, target, 1e-5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    describe("batch_size>1", () => {
      it(
        "default (top_k=1)",
        async () => {
          const output = await pipe(["a", "b c"]);
          const target = [
            { label: "LABEL_0", score: 0.5076976418495178 },
            { label: "LABEL_0", score: 0.5077522993087769 },
          ];
          compare(output, target, 1e-5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "custom (top_k=2)",
        async () => {
          const output = await pipe(["a", "b c"], { top_k: 2 });
          const target = [
            [
              { label: "LABEL_0", score: 0.5076976418495178 },
              { label: "LABEL_1", score: 0.49230238795280457 },
            ],
            [
              { label: "LABEL_0", score: 0.5077522993087769 },
              { label: "LABEL_1", score: 0.49224773049354553 },
            ],
          ];
          compare(output, target, 1e-5);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "multi_label_classification",
        async () => {
          const problem_type = pipe.model.config.problem_type;
          pipe.model.config.problem_type = "multi_label_classification";

          const output = await pipe(["a", "b c"], { top_k: 2 });
          const target = [
            [
              { label: "LABEL_0", score: 0.5001373887062073 },
              { label: "LABEL_1", score: 0.49243971705436707 },
            ],
            [
              { label: "LABEL_0", score: 0.5001326203346252 },
              { label: "LABEL_1", score: 0.492380291223526 },
            ],
          ];
          compare(output, target, 1e-5);

          // Reset problem type
          pipe.model.config.problem_type = problem_type;
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    afterAll(async () => {
      await pipe?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("token-classification", () => {
    const model_id = "hf-internal-testing/tiny-random-BertForTokenClassification";

    /** @type {TokenClassificationPipeline} */
    let pipe;
    beforeAll(async () => {
      pipe = await pipeline("token-classification", model_id, DEFAULT_MODEL_OPTIONS);
    }, MAX_MODEL_LOAD_TIME);

    describe("batch_size=1", () => {
      it(
        "default",
        async () => {
          const output = await pipe("1 2 3");

          // TODO: Add start/end to target
          const target = [
            {
              entity: "LABEL_0",
              score: 0.5292708,
              index: 1,
              word: "1",
              // 'start': 0, 'end': 1
            },
            {
              entity: "LABEL_0",
              score: 0.5353687,
              index: 2,
              word: "2",
              // 'start': 2, 'end': 3
            },
            {
              entity: "LABEL_1",
              score: 0.51381934,
              index: 3,
              word: "3",
              // 'start': 4, 'end': 5
            },
          ];
          compare(output, target, 1e-5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "custom (ignore_labels set)",
        async () => {
          const output = await pipe("1 2 3", { ignore_labels: ["LABEL_0"] });
          const target = [
            {
              entity: "LABEL_1",
              score: 0.51381934,
              index: 3,
              word: "3",
              // 'start': 4, 'end': 5
            },
          ];
          compare(output, target, 1e-5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    describe("batch_size>1", () => {
      it(
        "default",
        async () => {
          const output = await pipe(["1 2 3", "4 5"]);
          const target = [
            [
              {
                entity: "LABEL_0",
                score: 0.5292708,
                index: 1,
                word: "1",
                // 'start': 0, 'end': 1
              },
              {
                entity: "LABEL_0",
                score: 0.5353687,
                index: 2,
                word: "2",
                // 'start': 2, 'end': 3
              },
              {
                entity: "LABEL_1",
                score: 0.51381934,
                index: 3,
                word: "3",
                // 'start': 4, 'end': 5
              },
            ],
            [
              {
                entity: "LABEL_0",
                score: 0.5432807,
                index: 1,
                word: "4",
                // 'start': 0, 'end': 1
              },
              {
                entity: "LABEL_1",
                score: 0.5007693,
                index: 2,
                word: "5",
                // 'start': 2, 'end': 3
              },
            ],
          ];
          compare(output, target, 1e-5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "custom (ignore_labels set)",
        async () => {
          const output = await pipe(["1 2 3", "4 5"], { ignore_labels: ["LABEL_0"] });
          const target = [
            [
              {
                entity: "LABEL_1",
                score: 0.51381934,
                index: 3,
                word: "3",
                // 'start': 4, 'end': 5
              },
            ],
            [
              {
                entity: "LABEL_1",
                score: 0.5007693,
                index: 2,
                word: "5",
                // 'start': 2, 'end': 3
              },
            ],
          ];
          compare(output, target, 1e-5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    afterAll(async () => {
      await pipe?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("question-answering", () => {
    const model_id = "hf-internal-testing/tiny-random-BertForQuestionAnswering";

    /** @type {QuestionAnsweringPipeline} */
    let pipe;
    beforeAll(async () => {
      pipe = await pipeline("question-answering", model_id, DEFAULT_MODEL_OPTIONS);
    }, MAX_MODEL_LOAD_TIME);

    describe("batch_size=1", () => {
      it(
        "default (top_k=1)",
        async () => {
          const output = await pipe("a", "b c");
          const target = { score: 0.11395696550607681, /* start: 0, end: 1, */ answer: "b" };
          compare(output, target, 1e-5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "custom (top_k=3)",
        async () => {
          const output = await pipe("a", "b c", { top_k: 3 });
          const target = [
            { score: 0.11395696550607681, /* start: 0, end: 1, */ answer: "b" },
            { score: 0.11300431191921234, /* start: 2, end: 3, */ answer: "c" },
            { score: 0.10732574015855789, /* start: 0, end: 3, */ answer: "b c" },
          ];
          compare(output, target, 1e-5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    afterAll(async () => {
      await pipe?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("image-classification", () => {
    const model_id = "hf-internal-testing/tiny-random-vit";
    const urls = ["https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/white-image.png", "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/blue-image.png"];

    /** @type {ImageClassificationPipeline} */
    let pipe;
    beforeAll(async () => {
      pipe = await pipeline("image-classification", model_id, DEFAULT_MODEL_OPTIONS);
    }, MAX_MODEL_LOAD_TIME);

    describe("batch_size=1", () => {
      it(
        "default (top_k=5)",
        async () => {
          const output = await pipe(urls[0]);
          const target = [
            { label: "LABEL_1", score: 0.5020533800125122 },
            { label: "LABEL_0", score: 0.4979466497898102 },
          ];
          compare(output, target, 1e-5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "custom (top_k=1)",
        async () => {
          const output = await pipe(urls[0], { top_k: 1 });
          const target = [{ label: "LABEL_1", score: 0.5020533800125122 }];
          compare(output, target, 1e-5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    describe("batch_size>1", () => {
      it(
        "default (top_k=5)",
        async () => {
          const output = await pipe(urls);
          const target = [
            [
              { label: "LABEL_1", score: 0.5020533800125122 },
              { label: "LABEL_0", score: 0.4979466497898102 },
            ],
            [
              { label: "LABEL_1", score: 0.519227921962738 },
              { label: "LABEL_0", score: 0.4807720482349396 },
            ],
          ];
          compare(output, target, 1e-5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "custom (top_k=1)",
        async () => {
          const output = await pipe(urls, { top_k: 1 });
          const target = [[{ label: "LABEL_1", score: 0.5020533800125122 }], [{ label: "LABEL_1", score: 0.519227921962738 }]];
          compare(output, target, 1e-5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    afterAll(async () => {
      await pipe?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("zero-shot-image-classification", () => {
    const model_id = "hf-internal-testing/tiny-random-GroupViTModel";

    // Example adapted from https://huggingface.co/docs/transformers/en/model_doc/groupvit
    const urls = ["https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/white-image.png", "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/blue-image.png"];
    const labels = ["cat", "dog"];
    const hypothesis_template = "a photo of a {}";

    /** @type {ZeroShotImageClassificationPipeline} */
    let pipe;
    beforeAll(async () => {
      pipe = await pipeline("zero-shot-image-classification", model_id, DEFAULT_MODEL_OPTIONS);
    }, MAX_MODEL_LOAD_TIME);

    describe("batch_size=1", () => {
      it(
        "default",
        async () => {
          const output = await pipe(urls[0], labels);
          const target = [
            { score: 0.5990662574768066, label: "cat" },
            { score: 0.40093377232551575, label: "dog" },
          ];
          compare(output, target, 1e-5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "custom (w/ hypothesis_template)",
        async () => {
          const output = await pipe(urls[0], labels, { hypothesis_template });
          const target = [
            { score: 0.5527022480964661, label: "cat" },
            { score: 0.44729775190353394, label: "dog" },
          ];
          compare(output, target, 1e-5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    describe("batch_size>1", () => {
      it(
        "default",
        async () => {
          const output = await pipe(urls, labels);
          const target = [
            [
              { score: 0.5990662574768066, label: "cat" },
              { score: 0.40093377232551575, label: "dog" },
            ],
            [
              { score: 0.5006340146064758, label: "dog" },
              { score: 0.49936598539352417, label: "cat" },
            ],
          ];
          compare(output, target, 1e-5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "custom (w/ hypothesis_template)",
        async () => {
          const output = await pipe(urls, labels, { hypothesis_template });
          const target = [
            [
              { score: 0.5527022480964661, label: "cat" },
              { score: 0.44729775190353394, label: "dog" },
            ],
            [
              { score: 0.5395973324775696, label: "cat" },
              { score: 0.46040263772010803, label: "dog" },
            ],
          ];
          compare(output, target, 1e-5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    afterAll(async () => {
      await pipe?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("audio-classification", () => {
    const model_id = "hf-internal-testing/tiny-random-unispeech";
    const audios = [new Float32Array(16000).fill(0), Float32Array.from({ length: 16000 }, (_, i) => i)];

    /** @type {ImageClassificationPipeline} */
    let pipe;
    beforeAll(async () => {
      pipe = await pipeline("audio-classification", model_id, DEFAULT_MODEL_OPTIONS);
    }, MAX_MODEL_LOAD_TIME);

    describe("batch_size=1", () => {
      it(
        "default (top_k=5)",
        async () => {
          const output = await pipe(audios[0]);
          const target = [
            { score: 0.5043687224388123, label: "LABEL_0" },
            { score: 0.4956313371658325, label: "LABEL_1" },
          ];
          compare(output, target, 1e-5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "custom (top_k=1)",
        async () => {
          const output = await pipe(audios[0], { top_k: 1 });
          const target = [{ score: 0.5043687224388123, label: "LABEL_0" }];
          compare(output, target, 1e-5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    describe("batch_size>1", () => {
      it(
        "default (top_k=5)",
        async () => {
          const output = await pipe(audios);
          const target = [
            [
              { score: 0.5043687224388123, label: "LABEL_0" },
              { score: 0.4956313371658325, label: "LABEL_1" },
            ],
            [
              { score: 0.5187293887138367, label: "LABEL_0" },
              { score: 0.4812707006931305, label: "LABEL_1" },
            ],
          ];
          compare(output, target, 1e-5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "custom (top_k=1)",
        async () => {
          const output = await pipe(audios, { top_k: 1 });
          const target = [[{ score: 0.5043687224388123, label: "LABEL_0" }], [{ score: 0.5187293887138367, label: "LABEL_0" }]];
          compare(output, target, 1e-5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    afterAll(async () => {
      await pipe?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("text-generation", () => {
    const model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM";

    /** @type {TextGenerationPipeline} */
    let pipe;
    beforeAll(async () => {
      pipe = await pipeline("text-generation", model_id, DEFAULT_MODEL_OPTIONS);
    }, MAX_MODEL_LOAD_TIME);

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
          compare(output, text_target);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "text input (list)",
        async () => {
          const output = await pipe([text_input], { max_new_tokens: 3 });
          compare(output, [text_target]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "text input (single) - return_full_text=false",
        async () => {
          const output = await pipe(text_input, { max_new_tokens: 3, return_full_text: false });
          compare(output, new_text_target);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "text input (list) - return_full_text=false",
        async () => {
          const output = await pipe([text_input], { max_new_tokens: 3, return_full_text: false });
          compare(output, [new_text_target]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "chat input (single)",
        async () => {
          const output = await pipe(chat_input, { max_new_tokens: 3 });
          compare(output, chat_target);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "chat input (list)",
        async () => {
          const output = await pipe([chat_input], { max_new_tokens: 3 });
          compare(output, [chat_target]);
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
    //         compare(output, target);
    //     }, MAX_TEST_EXECUTION_TIME);
    // });

    afterAll(async () => {
      await pipe?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("translation", () => {
    const model_id = "Xenova/tiny-random-M2M100ForConditionalGeneration";

    /** @type {TranslationPipeline} */
    let pipe;
    beforeAll(async () => {
      pipe = await pipeline("translation", model_id, DEFAULT_MODEL_OPTIONS);
    }, MAX_MODEL_LOAD_TIME);

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
          compare(output, target);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    afterAll(async () => {
      await pipe?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("object-detection", () => {
    const model_id = "hf-internal-testing/tiny-random-DetrForObjectDetection";
    const urls = ["https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/white-image.png", "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/blue-image.png"];

    /** @type {ImageClassificationPipeline} */
    let pipe;
    beforeAll(async () => {
      pipe = await pipeline("object-detection", model_id, DEFAULT_MODEL_OPTIONS);
    }, MAX_MODEL_LOAD_TIME);

    describe("batch_size=1", () => {
      it(
        "default (threshold unset)",
        async () => {
          const output = await pipe(urls[0]);
          const target = [];
          compare(output, target, 1e-5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "default (threshold=0)",
        async () => {
          const output = await pipe(urls[0], { threshold: 0 });
          const target = [
            { score: 0.020360443741083145, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
            { score: 0.020360419526696205, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
            { score: 0.02036038413643837, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
            { score: 0.020360447466373444, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
            { score: 0.020360389724373817, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
            { score: 0.020360423251986504, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
            { score: 0.02036040835082531, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
            { score: 0.020360363647341728, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
            { score: 0.020360389724373817, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
            { score: 0.020360389724373817, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
            { score: 0.020360343158245087, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
            { score: 0.020360423251986504, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
          ];
          compare(output, target, 1e-5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    // TODO: Add batched support to object detection pipeline
    // describe('batch_size>1', () => {
    //     it('default (threshold unset)', async () => {
    //         const output = await pipe(urls);
    //         console.log(output);
    //         const target = [];
    //         compare(output, target, 1e-5);
    //     }, MAX_TEST_EXECUTION_TIME);
    //     it('default (threshold=0)', async () => {
    //         const output = await pipe(urls, { threshold: 0 });
    //         console.log(output);
    //         const target = [];
    //         compare(output, target, 1e-5);
    //     }, MAX_TEST_EXECUTION_TIME);
    // });

    afterAll(async () => {
      await pipe?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("document-question-answering", () => {
    const model_id = "hf-internal-testing/tiny-random-VisionEncoderDecoderModel-donutswin-mbart";

    /** @type {DocumentQuestionAnsweringPipeline} */
    let pipe;
    beforeAll(async () => {
      pipe = await pipeline("document-question-answering", model_id, DEFAULT_MODEL_OPTIONS);
    }, MAX_MODEL_LOAD_TIME);

    describe("batch_size=1", () => {
      it(
        "default",
        async () => {
          const dims = [64, 32, 3];
          const image = new RawImage(new Uint8ClampedArray(dims[0] * dims[1] * dims[2]).fill(255), ...dims);
          const question = "What is the invoice number?";
          const output = await pipe(image, question);

          const target = [{ answer: null }];
          compare(output, target);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    afterAll(async () => {
      await pipe?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
});
