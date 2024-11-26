import {
  // Tokenizers
  CodeGenTokenizer,
  LlamaTokenizer,
  CohereTokenizer,
  GemmaTokenizer,
  GPT2Tokenizer,
  GPTNeoXTokenizer,
  BloomTokenizer,
  BertTokenizer,
  T5Tokenizer,
  WhisperTokenizer,
  BartTokenizer,
  MarianTokenizer,
  PreTrainedTokenizer,
  AutoTokenizer,

  // Processors
  CLIPImageProcessor,
  AutoProcessor,
  Processor,
  Florence2Processor,

  // Models
  LlamaForCausalLM,
  OlmoForCausalLM,
  GraniteForCausalLM,
  CohereModel,
  CohereForCausalLM,
  GemmaForCausalLM,
  Gemma2ForCausalLM,
  OPTForCausalLM,
  GPTNeoXForCausalLM,
  GPTJForCausalLM,
  BloomForCausalLM,
  GPTBigCodeForCausalLM,
  GPT2LMHeadModel,
  JAISLMHeadModel,
  MptForCausalLM,
  CodeGenForCausalLM,
  MistralForCausalLM,
  GPTNeoForCausalLM,
  BertForMaskedLM,
  BertForSequenceClassification,
  T5ForConditionalGeneration,
  T5Model,
  BertModel,
  BertForTokenClassification,
  BertForQuestionAnswering,
  MusicgenForConditionalGeneration,
  LlavaForConditionalGeneration,
  WhisperForConditionalGeneration,
  VisionEncoderDecoderModel,
  Florence2ForConditionalGeneration,
  Qwen2VLForConditionalGeneration,
  MarianMTModel,
  PatchTSTModel,
  PatchTSTForPrediction,
  PatchTSMixerModel,
  PatchTSMixerForPrediction,

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
  full,
  RawImage,
  Tensor,
} from "../src/transformers.js";

import { init, MAX_TEST_TIME, MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME } from "./init.js";
import { compare } from "./test_utils.js";

init();

const DEFAULT_MODEL_OPTIONS = {
  dtype: "fp32",
};
describe("Tiny random models", () => {
  describe("bert", () => {
    describe("BertModel", () => {
      const model_id = "hf-internal-testing/tiny-random-BertModel";

      /** @type {BertModel} */
      let model;
      /** @type {BertTokenizer} */
      let tokenizer;
      beforeAll(async () => {
        model = await BertModel.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
        tokenizer = await BertTokenizer.from_pretrained(model_id);
      }, MAX_MODEL_LOAD_TIME);

      it(
        "batch_size=1",
        async () => {
          const inputs = tokenizer("hello");
          const { last_hidden_state } = await model(inputs);
          expect(last_hidden_state.dims).toEqual([1, 7, 32]);
          expect(last_hidden_state.mean().item()).toBeCloseTo(0.0, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "batch_size>1",
        async () => {
          const inputs = tokenizer(["hello", "hello world"], { padding: true });
          const { last_hidden_state } = await model(inputs);
          expect(last_hidden_state.dims).toEqual([2, 12, 32]);
          expect(last_hidden_state.mean().item()).toBeCloseTo(1.4901161193847656e-8, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await model?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });

    describe("BertForMaskedLM", () => {
      const model_id = "hf-internal-testing/tiny-random-BertForMaskedLM";

      const texts = ["The goal of life is [MASK].", "Paris is the [MASK] of France."];

      /** @type {BertForMaskedLM} */
      let model;
      /** @type {BertTokenizer} */
      let tokenizer;
      beforeAll(async () => {
        model = await BertForMaskedLM.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
        tokenizer = await BertTokenizer.from_pretrained(model_id);
      }, MAX_MODEL_LOAD_TIME);

      it(
        "batch_size=1",
        async () => {
          const inputs = tokenizer(texts[0]);
          const { logits } = await model(inputs);
          expect(logits.dims).toEqual([1, 19, 1124]);
          expect(logits.mean().item()).toBeCloseTo(0.0016587056452408433, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "batch_size>1",
        async () => {
          const inputs = tokenizer(texts, { padding: true });
          const { logits } = await model(inputs);
          expect(logits.dims).toEqual([2, 22, 1124]);
          expect(logits.mean().item()).toBeCloseTo(0.0017160633578896523, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await model?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });

    describe("BertForSequenceClassification", () => {
      const model_id = "hf-internal-testing/tiny-random-BertForSequenceClassification";

      /** @type {BertForSequenceClassification} */
      let model;
      /** @type {BertTokenizer} */
      let tokenizer;
      beforeAll(async () => {
        model = await BertForSequenceClassification.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
        tokenizer = await BertTokenizer.from_pretrained(model_id);
      }, MAX_MODEL_LOAD_TIME);

      it(
        "batch_size=1",
        async () => {
          const inputs = tokenizer("hello");
          const { logits } = await model(inputs);
          const target = [[0.00043986947275698185, -0.030218850821256638]].flat();
          expect(logits.dims).toEqual([1, 2]);
          logits
            .tolist()
            .flat()
            .forEach((item, i) => {
              expect(item).toBeCloseTo(target[i], 5);
            });
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "batch_size>1",
        async () => {
          const inputs = tokenizer(["hello", "hello world"], { padding: true });
          const { logits } = await model(inputs);
          const target = [
            [0.00043986947275698185, -0.030218850821256638],
            [0.0003853091038763523, -0.03022204339504242],
          ].flat();
          expect(logits.dims).toEqual([2, 2]);
          logits
            .tolist()
            .flat()
            .forEach((item, i) => {
              expect(item).toBeCloseTo(target[i], 5);
            });
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await model?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });

    describe("BertForTokenClassification", () => {
      const model_id = "hf-internal-testing/tiny-random-BertForTokenClassification";

      /** @type {BertForTokenClassification} */
      let model;
      /** @type {BertTokenizer} */
      let tokenizer;
      beforeAll(async () => {
        model = await BertForTokenClassification.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
        tokenizer = await BertTokenizer.from_pretrained(model_id);
      }, MAX_MODEL_LOAD_TIME);

      it(
        "batch_size=1",
        async () => {
          const inputs = tokenizer("hello");
          const { logits } = await model(inputs);
          expect(logits.dims).toEqual([1, 7, 2]);
          expect(logits.mean().item()).toBeCloseTo(0.07089076191186905, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "batch_size>1",
        async () => {
          const inputs = tokenizer(["hello", "hello world"], { padding: true });
          const { logits } = await model(inputs);
          expect(logits.dims).toEqual([2, 12, 2]);
          expect(logits.mean().item()).toBeCloseTo(0.04702216014266014, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await model?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });

    describe("BertForQuestionAnswering", () => {
      const model_id = "hf-internal-testing/tiny-random-BertForQuestionAnswering";

      /** @type {BertForQuestionAnswering} */
      let model;
      /** @type {BertTokenizer} */
      let tokenizer;
      beforeAll(async () => {
        model = await BertForQuestionAnswering.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
        tokenizer = await BertTokenizer.from_pretrained(model_id);
      }, MAX_MODEL_LOAD_TIME);

      it(
        "batch_size=1",
        async () => {
          const inputs = tokenizer("hello");
          const { start_logits, end_logits } = await model(inputs);
          expect(start_logits.dims).toEqual([1, 7]);
          expect(start_logits.mean().item()).toBeCloseTo(0.12772157788276672, 5);
          expect(end_logits.dims).toEqual([1, 7]);
          expect(end_logits.mean().item()).toBeCloseTo(0.11811424791812897, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "batch_size>1",
        async () => {
          const inputs = tokenizer(["hello", "hello world"], { padding: true });
          const { start_logits, end_logits } = await model(inputs);
          expect(start_logits.dims).toEqual([2, 12]);
          expect(start_logits.mean().item()).toBeCloseTo(0.12843115627765656, 5);
          expect(end_logits.dims).toEqual([2, 12]);
          expect(end_logits.mean().item()).toBeCloseTo(0.11745202541351318, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await model?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });
  });

  describe("t5", () => {
    describe("T5Model", () => {
      const model_id = "hf-internal-testing/tiny-random-T5Model";

      /** @type {T5Model} */
      let model;
      /** @type {T5Tokenizer} */
      let tokenizer;
      beforeAll(async () => {
        model = await T5Model.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
        tokenizer = await T5Tokenizer.from_pretrained(model_id);
      }, MAX_MODEL_LOAD_TIME);

      it(
        "forward",
        async () => {
          // Example adapted from https://huggingface.co/google-t5/t5-small#how-to-get-started-with-the-model
          const inputs = tokenizer("Studies have been shown that owning a dog is good for you");
          const { input_ids: decoder_input_ids } = tokenizer("Studies show that");

          const { last_hidden_state } = await model({ ...inputs, decoder_input_ids });
          expect(last_hidden_state.dims).toEqual([1, 4, 32]);
          expect(last_hidden_state.mean().item()).toBeCloseTo(7.492632721550763e-5, 8);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await model?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });
    describe("T5ForConditionalGeneration", () => {
      const model_id = "hf-internal-testing/tiny-random-T5ForConditionalGeneration";

      /** @type {T5ForConditionalGeneration} */
      let model;
      /** @type {T5Tokenizer} */
      let tokenizer;
      beforeAll(async () => {
        model = await T5ForConditionalGeneration.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
        tokenizer = await T5Tokenizer.from_pretrained(model_id);
      }, MAX_MODEL_LOAD_TIME);

      it(
        "forward",
        async () => {
          // Example adapted from https://huggingface.co/google-t5/t5-small#how-to-get-started-with-the-model
          const inputs = tokenizer("Studies have been shown that owning a dog is good for you");
          const { input_ids: decoder_input_ids } = tokenizer("Studies show that");

          const model = await T5ForConditionalGeneration.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
          const outputs = await model({ ...inputs, decoder_input_ids });
          expect(outputs.logits.dims).toEqual([1, 4, 32100]);
          expect(outputs.logits.mean().item()).toBeCloseTo(8.867568901393952e-9, 12);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "batch_size=1",
        async () => {
          const inputs = tokenizer("hello");
          const outputs = await model.generate({
            ...inputs,
            max_length: 10,
          });
          expect(outputs.tolist()).toEqual([[0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n]]);
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
            [0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n],
            [0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n],
          ]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await model?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });
  });

  describe("marian", () => {
    describe("MarianMTModel", () => {
      const model_id = "onnx-community/tiny-random-MarianMTModel";

      /** @type {MarianMTModel} */
      let model;
      /** @type {MarianTokenizer} */
      let tokenizer;
      beforeAll(async () => {
        model = await MarianMTModel.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
        tokenizer = await MarianTokenizer.from_pretrained(model_id);
      }, MAX_MODEL_LOAD_TIME);

      it(
        "batch_size=1",
        async () => {
          const inputs = tokenizer("hello");
          const outputs = await model.generate({
            ...inputs,
            max_length: 10,
          });
          expect(outputs.tolist()).toEqual([[3n, 40672n, 8358n, 32810n, 32810n, 32810n, 32810n, 35687n, 33073n, 6870n]]);
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
            [3n, 40672n, 8358n, 32810n, 32810n, 32810n, 32810n, 35687n, 33073n, 6870n],
            [3n, 40672n, 8358n, 32810n, 32810n, 32810n, 32810n, 35687n, 33073n, 6870n],
          ]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await model?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });
  });

  describe("musicgen", () => {
    describe("MusicgenForConditionalGeneration", () => {
      const model_id = "hf-internal-testing/tiny-random-MusicgenForConditionalGeneration";

      // Example adapted from https://huggingface.co/docs/transformers/model_doc/musicgen#text-conditional-generation
      const texts = ["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"];

      /** @type {MusicgenForConditionalGeneration} */
      let model;
      /** @type {T5Tokenizer} */
      let tokenizer;
      beforeAll(async () => {
        model = await MusicgenForConditionalGeneration.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
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
  });

  describe("whisper", () => {
    describe("WhisperForConditionalGeneration", () => {
      const model_id = "Xenova/tiny-random-WhisperForConditionalGeneration";

      /** @type {WhisperForConditionalGeneration} */
      let model;
      /** @type {WhisperTokenizer} */
      let tokenizer;
      beforeAll(async () => {
        model = await WhisperForConditionalGeneration.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
        tokenizer = await WhisperTokenizer.from_pretrained(model_id);
      }, MAX_MODEL_LOAD_TIME);

      describe("prefix tokens", () => {
        const input_features = full([1, 80, 3000], 0.0);

        describe("English-only", () => {
          it(
            "default",
            async () => {
              const outputs = await model.generate({
                input_features,
                is_multilingual: false,
                max_new_tokens: 1,
              });

              expect(outputs.tolist()).toEqual([[/* Prefix */ 50258n, 50363n, /* Generated */ 45084n]]);
            },
            MAX_TEST_EXECUTION_TIME,
          );

          it(
            "return_timestamps=true",
            async () => {
              const outputs = await model.generate({
                input_features,
                is_multilingual: false,
                max_new_tokens: 1,
                return_timestamps: true,
              });

              expect(outputs.tolist()).toEqual([[/* Prefix */ 50258n, /* Generated */ 50366n]]);
            },
            MAX_TEST_EXECUTION_TIME,
          );
        });

        describe("multilingual", () => {
          it(
            "language unset; task unset",
            async () => {
              // language defaults to 'en'
              // task defaults to 'transcribe'

              const outputs = await model.generate({
                input_features,
                max_new_tokens: 1,
              });

              expect(outputs.tolist()).toEqual([[/* Prefix */ 50258n, 50259n, 50359n, 50363n, /* Generated */ 45084n]]);
            },
            MAX_TEST_EXECUTION_TIME,
          );

          it(
            "language set; task unset",
            async () => {
              // task defaults to 'transcribe'
              const outputs = await model.generate({
                input_features,
                max_new_tokens: 1,
                language: "af",
              });

              expect(outputs.tolist()).toEqual([[/* Prefix */ 50258n, 50327n, 50359n, 50363n, /* Generated */ 45084n]]);
            },
            MAX_TEST_EXECUTION_TIME,
          );

          it(
            "language set; task set",
            async () => {
              const outputs = await model.generate({
                input_features,
                max_new_tokens: 1,
                language: "zh",
                task: "translate",
              });

              expect(outputs.tolist()).toEqual([[/* Prefix */ 50258n, 50260n, 50358n, 50363n, /* Generated */ 45084n]]);
            },
            MAX_TEST_EXECUTION_TIME,
          );

          it(
            "return_timestamps=true",
            async () => {
              const outputs = await model.generate({
                input_features,
                max_new_tokens: 1,
                language: "en",
                task: "transcribe",
                return_timestamps: true,
              });

              expect(outputs.tolist()).toEqual([[/* Prefix */ 50258n, 50259n, 50359n, /* Generated */ 50400n]]);
            },
            MAX_TEST_EXECUTION_TIME,
          );
        });
      });

      describe("decoder_start_ids", () => {
        const input_features = full([1, 80, 3000], 0.0);

        it(
          "broadcast inputs",
          async () => {
            const { decoder_start_token_id, lang_to_id, task_to_id, no_timestamps_token_id } = model.generation_config;

            const outputs = await model.generate({
              input_features, // batch size 1
              max_new_tokens: 1,
              decoder_input_ids: [
                // batch size 2
                // <|startoftranscript|> <|lang_id|> <|task|> [<|notimestamps|>]
                [decoder_start_token_id, lang_to_id["<|en|>"], task_to_id["translate"], no_timestamps_token_id],
                [decoder_start_token_id, lang_to_id["<|fr|>"], task_to_id["transcribe"], no_timestamps_token_id],
              ],
            });
            expect(outputs.tolist()).toEqual([
              [/* Prefix */ 50258n, 50259n, 50358n, 50363n, /* Generated */ 45084n],
              [/* Prefix */ 50258n, 50265n, 50359n, 50363n, /* Generated */ 45084n],
            ]);
          },
          MAX_TEST_EXECUTION_TIME,
        );
      });

      afterAll(async () => {
        await model?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });
  });

  describe("llava", () => {
    const prompts = [
      // Example adapted from https://huggingface.co/docs/transformers/model_doc/llava#transformers.LlavaForConditionalGeneration.forward.example
      "<image>\nUSER: What's the content of the image?\nASSISTANT:",
      "<image>Hi",
    ];

    // Empty white image
    const dims = [224, 224, 3];
    const image = new RawImage(new Uint8ClampedArray(dims[0] * dims[1] * dims[2]).fill(255), ...dims);

    describe("LlavaForConditionalGeneration", () => {
      const model_id = "Xenova/tiny-random-LlavaForConditionalGeneration";

      /** @type {LlavaForConditionalGeneration} */
      let model;
      /** @type {LlamaTokenizer} */
      let tokenizer;
      /** @type {CLIPImageProcessor} */
      let processor;
      beforeAll(async () => {
        model = await LlavaForConditionalGeneration.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
        tokenizer = await LlamaTokenizer.from_pretrained(model_id);
        processor = await AutoProcessor.from_pretrained(model_id);
      }, MAX_MODEL_LOAD_TIME);

      it(
        "forward",
        async () => {
          const text_inputs = tokenizer(prompts[0]);
          const vision_inputs = await processor(image);
          const inputs = { ...text_inputs, ...vision_inputs };

          const { logits } = await model(inputs);
          expect(logits.dims).toEqual([1, 244, 32002]);
          expect(logits.mean().item()).toBeCloseTo(-0.0005755752790719271, 8);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "batch_size=1",
        async () => {
          const text_inputs = tokenizer(prompts[0]);
          const vision_inputs = await processor(image);
          const inputs = { ...text_inputs, ...vision_inputs };

          const generate_ids = await model.generate({ ...inputs, max_new_tokens: 10 });
          expect(generate_ids.tolist()).toEqual([[1n, 32000n, 29871n, 13n, 11889n, 29901n, 1724n, 29915n, 29879n, 278n, 2793n, 310n, 278n, 1967n, 29973n, 13n, 22933n, 9047n, 13566n, 29901n, 21557n, 16781n, 27238n, 8279n, 20454n, 11927n, 12462n, 12306n, 2414n, 7561n]]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "batch_size>1",
        async () => {
          const text_inputs = tokenizer(prompts, { padding: true });
          const vision_inputs = await processor([image, image]);
          const inputs = { ...text_inputs, ...vision_inputs };

          const generate_ids = await model.generate({ ...inputs, max_new_tokens: 10 });
          expect(generate_ids.tolist()).toEqual([
            [1n, 32000n, 29871n, 13n, 11889n, 29901n, 1724n, 29915n, 29879n, 278n, 2793n, 310n, 278n, 1967n, 29973n, 13n, 22933n, 9047n, 13566n, 29901n, 21557n, 16781n, 27238n, 8279n, 20454n, 11927n, 12462n, 12306n, 2414n, 7561n],
            [0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 0n, 1n, 32000n, 6324n, 1217n, 22958n, 22913n, 10381n, 148n, 31410n, 31736n, 7358n, 9150n, 28635n],
          ]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await model?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });
  });

  describe("florence2", () => {
    const texts = ["Describe with a paragraph what is shown in the image.", "Locate the objects with category name in the image."];

    // Empty white image
    const dims = [224, 224, 3];
    const image = new RawImage(new Uint8ClampedArray(dims[0] * dims[1] * dims[2]).fill(255), ...dims);

    describe("Florence2ForConditionalGeneration", () => {
      const model_id = "Xenova/tiny-random-Florence2ForConditionalGeneration";

      /** @type {Florence2ForConditionalGeneration} */
      let model;
      /** @type {Florence2Processor} */
      let processor;
      beforeAll(async () => {
        model = await Florence2ForConditionalGeneration.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
        processor = await AutoProcessor.from_pretrained(model_id);
      }, MAX_MODEL_LOAD_TIME);

      it(
        "forward",
        async () => {
          const inputs = await processor(image, texts[0]);

          const { logits } = await model({
            ...inputs,
            decoder_input_ids: full([1, 1], 2n),
          });
          expect(logits.dims).toEqual([1, 1, 51289]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "batch_size=1",
        async () => {
          {
            const text_inputs = processor.tokenizer(texts[0]);
            const generate_ids = await model.generate({ ...text_inputs, max_new_tokens: 10 });
            expect(generate_ids.tolist()).toEqual([[2n, 0n, 0n, 0n, 1n, 0n, 0n, 2n]]);
          }
          {
            const inputs = await processor(image, texts[0]);
            const generate_ids = await model.generate({ ...inputs, max_new_tokens: 10 });
            expect(generate_ids.tolist()).toEqual([[2n, 0n, 48n, 48n, 48n, 48n, 48n, 48n, 48n, 48n, 2n]]);
          }
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "batch_size>1",
        async () => {
          {
            const text_inputs = processor.tokenizer(texts, { padding: true });
            const generate_ids = await model.generate({ ...text_inputs, max_new_tokens: 10 });
            expect(generate_ids.tolist()).toEqual([
              [2n, 0n, 0n, 0n, 1n, 0n, 0n, 2n],
              [2n, 0n, 0n, 0n, 1n, 0n, 0n, 2n],
            ]);
          }
          {
            const inputs = await processor([image, image], texts, { padding: true });

            const generate_ids = await model.generate({ ...inputs, max_new_tokens: 10 });
            expect(generate_ids.tolist()).toEqual([
              [2n, 0n, 48n, 48n, 48n, 48n, 48n, 48n, 48n, 48n, 2n],
              [2n, 0n, 48n, 48n, 48n, 48n, 48n, 48n, 48n, 48n, 2n],
            ]);
          }
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await model?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });
  });

  describe("qwen2_vl", () => {
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

    describe("Qwen2VLForConditionalGeneration", () => {
      const model_id = "hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration";

      /** @type {Qwen2VLForConditionalGeneration} */
      let model;
      /** @type {Qwen2VLProcessor} */
      let processor;
      beforeAll(async () => {
        model = await Qwen2VLForConditionalGeneration.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
        processor = await AutoProcessor.from_pretrained(model_id);
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
  });

  describe("vision-encoder-decoder", () => {
    describe("VisionEncoderDecoderModel", () => {
      const model_id = "hf-internal-testing/tiny-random-VisionEncoderDecoderModel-vit-gpt2";

      /** @type {VisionEncoderDecoderModel} */
      let model;
      /** @type {GPT2Tokenizer} */
      let tokenizer;
      beforeAll(async () => {
        model = await VisionEncoderDecoderModel.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
        tokenizer = await GPT2Tokenizer.from_pretrained(model_id);
      }, MAX_MODEL_LOAD_TIME);

      it(
        "batch_size=1",
        async () => {
          const outputs = await model.generate({
            pixel_values: full([1, 3, 30, 30], -1.0),
            max_length: 5,
          });
          expect(outputs.tolist()).toEqual([[0n, 400n, 400n, 400n, 400n]]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      // TODO: Add back
      // it('batch_size>1', async () => {
      //     const outputs = await model.generate({
      //         pixel_values: cat([
      //             full([1, 3, 30, 30], -1.0),
      //             full([1, 3, 30, 30], 0.0),
      //         ]),
      //         max_length: 5,
      //     });
      //     expect(outputs.tolist()).toEqual([
      //         // Generation continues
      //         [0n, 400n, 400n, 400n, 400n],

      //         // Finishes early. 1023 is the padding token
      //         [0n, 0n, 1023n, 1023n, 1023n],
      //     ]);
      // }, MAX_TEST_EXECUTION_TIME);

      afterAll(async () => {
        await model?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });
  });
  describe("opt", () => {
    describe("OPTForCausalLM", () => {
      const model_id = "hf-internal-testing/tiny-random-OPTForCausalLM";
      /** @type {OPTForCausalLM} */
      let model;
      /** @type {GPT2Tokenizer} */
      let tokenizer;
      beforeAll(async () => {
        model = await OPTForCausalLM.from_pretrained(model_id, {
          // TODO move to config
          revision: "refs/pr/2",
          ...DEFAULT_MODEL_OPTIONS,
        });
        tokenizer = await GPT2Tokenizer.from_pretrained(model_id, {
          // TODO update this
          revision: "refs/pr/3",
        });
        tokenizer.padding_side = "left";
      }, MAX_MODEL_LOAD_TIME);

      it(
        "batch_size=1",
        async () => {
          const inputs = tokenizer("hello");
          const outputs = await model.generate({
            ...inputs,
            max_length: 10,
          });
          expect(outputs.tolist()).toEqual([[2n, 42891n, 39144n, 39144n, 39144n, 39144n, 39144n, 39144n, 39144n, 39144n]]);
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
            [1n, 2n, 42891n, 39144n, 39144n, 39144n, 39144n, 39144n, 39144n, 39144n],
            [2n, 42891n, 232n, 24680n, 24680n, 24680n, 24680n, 24680n, 24680n, 24680n],
          ]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await model?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });
  });

  describe("llama", () => {
    describe("LlamaForCausalLM", () => {
      const model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM";
      /** @type {LlamaForCausalLM} */
      let model;
      /** @type {LlamaTokenizer} */
      let tokenizer;
      beforeAll(async () => {
        model = await LlamaForCausalLM.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
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
              // TODO move to config
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
  });

  describe("olmo", () => {
    describe("OlmoForCausalLM", () => {
      const model_id = "onnx-community/tiny-random-olmo-hf";
      /** @type {OlmoForCausalLM} */
      let model;
      /** @type {GPTNeoXTokenizer} */
      let tokenizer;
      beforeAll(async () => {
        model = await OlmoForCausalLM.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
        tokenizer = await GPTNeoXTokenizer.from_pretrained(model_id);
        tokenizer.padding_side = "left";
      }, MAX_MODEL_LOAD_TIME);

      it(
        "batch_size=1",
        async () => {
          const inputs = tokenizer("hello");
          const outputs = await model.generate({
            ...inputs,
            max_length: 10,
          });
          expect(outputs.tolist()).toEqual([[25521n, 10886n, 44936n, 38777n, 33038n, 18557n, 1810n, 33853n, 9517n, 28892n]]);
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
            [1n, 25521n, 10886n, 44936n, 38777n, 33038n, 18557n, 1810n, 33853n, 9517n],
            [25521n, 1533n, 37199n, 27362n, 30594n, 39261n, 8824n, 19175n, 8545n, 29335n],
          ]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await model?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });
  });

  describe("granite", () => {
    describe("GraniteForCausalLM", () => {
      const model_id = "hf-internal-testing/tiny-random-GraniteForCausalLM";
      /** @type {GraniteForCausalLM} */
      let model;
      /** @type {GPT2Tokenizer} */
      let tokenizer;
      beforeAll(async () => {
        model = await GraniteForCausalLM.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
        tokenizer = await GPT2Tokenizer.from_pretrained(model_id);
      }, MAX_MODEL_LOAD_TIME);

      it(
        "batch_size=1",
        async () => {
          const inputs = tokenizer("hello");
          const outputs = await model.generate({
            ...inputs,
            max_length: 10,
          });
          expect(outputs.tolist()).toEqual([[7656n, 39727n, 33077n, 9643n, 30539n, 47869n, 48739n, 15085n, 9203n, 14020n]]);
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
            [0n, 7656n, 39727n, 33077n, 9643n, 30539n, 47869n, 48739n, 15085n, 9203n],
            [7656n, 5788n, 17835n, 13234n, 7592n, 21471n, 30537n, 23023n, 43450n, 4824n],
          ]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await model?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });
  });

  describe("cohere", () => {
    describe("CohereModel", () => {
      const model_id = "hf-internal-testing/tiny-random-CohereModel";
      /** @type {CohereModel} */
      let model;
      /** @type {CohereTokenizer} */
      let tokenizer;
      beforeAll(async () => {
        model = await CohereModel.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
        tokenizer = await CohereTokenizer.from_pretrained(model_id);
        tokenizer.padding_side = "left";
      }, MAX_MODEL_LOAD_TIME);

      it(
        "batch_size=1",
        async () => {
          const inputs = tokenizer("hello");
          const { last_hidden_state } = await model(inputs);
          expect(last_hidden_state.dims).toEqual([1, 4, 32]);
          expect(last_hidden_state.mean().item()).toBeCloseTo(0.0, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "batch_size>1",
        async () => {
          const inputs = tokenizer(["hello", "hello world"], { padding: true });
          const { last_hidden_state } = await model(inputs);
          expect(last_hidden_state.dims).toEqual([2, 6, 32]);
          expect(last_hidden_state.mean().item()).toBeCloseTo(9.934107758624577e-9, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await model?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });

    describe("CohereForCausalLM", () => {
      const model_id = "hf-internal-testing/tiny-random-CohereForCausalLM";
      /** @type {CohereForCausalLM} */
      let model;
      /** @type {CohereTokenizer} */
      let tokenizer;
      beforeAll(async () => {
        model = await CohereForCausalLM.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
        tokenizer = await CohereTokenizer.from_pretrained(model_id);
        tokenizer.padding_side = "left";
      }, MAX_MODEL_LOAD_TIME);

      it(
        "batch_size=1",
        async () => {
          const inputs = tokenizer("hello");
          const outputs = await model.generate({
            ...inputs,
            max_length: 10,
          });
          expect(outputs.tolist()).toEqual([[5n, 203n, 790n, 87n, 87n, 87n, 87n, 87n, 87n, 87n]]);
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
            [0n, 0n, 5n, 203n, 790n, 87n, 87n, 87n, 87n, 87n],
            [5n, 203n, 790n, 87n, 214n, 741n, 741n, 741n, 741n, 741n],
          ]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await model?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });
  });

  describe("gemma", () => {
    describe("GemmaForCausalLM", () => {
      const model_id = "Xenova/tiny-random-GemmaForCausalLM";
      /** @type {GemmaForCausalLM} */
      let model;
      /** @type {GemmaTokenizer} */
      let tokenizer;
      beforeAll(async () => {
        model = await GemmaForCausalLM.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
        tokenizer = await GemmaTokenizer.from_pretrained(model_id);
        tokenizer.padding_side = "left";
      }, MAX_MODEL_LOAD_TIME);

      it(
        "batch_size=1",
        async () => {
          const inputs = tokenizer("hello");
          const outputs = await model.generate({
            ...inputs,
            max_length: 10,
          });
          expect(outputs.tolist()).toEqual([[2n, 17534n, 254059n, 254059n, 254059n, 254059n, 254059n, 254059n, 254059n, 254059n]]);
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
            [0n, 2n, 17534n, 254059n, 254059n, 254059n, 254059n, 254059n, 254059n, 254059n],
            [2n, 17534n, 2134n, 71055n, 71055n, 71055n, 71055n, 71055n, 71055n, 71055n],
          ]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await model?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });
  });

  describe("gemma", () => {
    describe("Gemma2ForCausalLM", () => {
      const model_id = "hf-internal-testing/tiny-random-Gemma2ForCausalLM";
      /** @type {Gemma2ForCausalLM} */
      let model;
      /** @type {GemmaTokenizer} */
      let tokenizer;
      beforeAll(async () => {
        model = await Gemma2ForCausalLM.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
        tokenizer = await GemmaTokenizer.from_pretrained(model_id);
        tokenizer.padding_side = "left";
      }, MAX_MODEL_LOAD_TIME);

      it(
        "batch_size=1",
        async () => {
          const inputs = tokenizer("hello");
          const outputs = await model.generate({
            ...inputs,
            max_length: 10,
          });
          expect(outputs.tolist()).toEqual([[2n, 17534n, 127534n, 160055n, 160055n, 160055n, 160055n, 160055n, 160055n, 160055n]]);
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
            [0n, 2n, 17534n, 127534n, 127534n, 215341n, 215341n, 215341n, 215341n, 215341n],
            [2n, 17534n, 2134n, 107508n, 160055n, 160055n, 160055n, 160055n, 160055n, 160055n],
          ]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await model?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });
  });

  describe("gpt_neo", () => {
    describe("GPTNeoForCausalLM", () => {
      const model_id = "hf-internal-testing/tiny-random-GPTNeoForCausalLM";
      /** @type {GPTNeoForCausalLM} */
      let model;
      /** @type {GPT2Tokenizer} */
      let tokenizer;
      beforeAll(async () => {
        model = await GPTNeoForCausalLM.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
        tokenizer = await GPT2Tokenizer.from_pretrained(model_id);
        tokenizer.padding_side = "left";
      }, MAX_MODEL_LOAD_TIME);

      it(
        "batch_size=1",
        async () => {
          const inputs = tokenizer("hello");
          const outputs = await model.generate({
            ...inputs,
            max_length: 10,
          });
          expect(outputs.tolist()).toEqual([[258n, 863n, 79n, 79n, 79n, 949n, 949n, 949n, 949n, 949n]]);
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
            [0n, 0n, 258n, 863n, 79n, 79n, 79n, 949n, 949n, 949n],
            [258n, 863n, 79n, 269n, 813n, 849n, 849n, 849n, 849n, 849n],
          ]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await model?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });
  });

  describe("gpt_neox", () => {
    describe("GPTNeoXForCausalLM", () => {
      const model_id = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM";
      /** @type {GPTNeoXForCausalLM} */
      let model;
      /** @type {GPTNeoXTokenizer} */
      let tokenizer;
      beforeAll(async () => {
        model = await GPTNeoXForCausalLM.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
        tokenizer = await GPTNeoXTokenizer.from_pretrained(model_id);
        tokenizer.padding_side = "left";
      }, MAX_MODEL_LOAD_TIME);

      it(
        "batch_size=1",
        async () => {
          const inputs = tokenizer("hello");
          const outputs = await model.generate({
            ...inputs,
            max_length: 10,
          });
          expect(outputs.tolist()).toEqual([[259n, 864n, 80n, 881n, 502n, 895n, 938n, 668n, 502n, 895n]]);
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
            [0n, 0n, 259n, 864n, 80n, 881n, 502n, 895n, 938n, 668n],
            [259n, 864n, 80n, 270n, 814n, 522n, 112n, 268n, 503n, 468n],
          ]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await model?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });
  });

  describe("gptj", () => {
    describe("GPTJForCausalLM", () => {
      const model_id = "hf-internal-testing/tiny-random-GPTJForCausalLM";
      /** @type {GPTJForCausalLM} */
      let model;
      /** @type {GPTNeoXTokenizer} */
      let tokenizer;
      beforeAll(async () => {
        model = await GPTJForCausalLM.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
        tokenizer = await GPTNeoXTokenizer.from_pretrained(model_id);
        tokenizer.padding_side = "left";
      }, MAX_MODEL_LOAD_TIME);

      it(
        "batch_size=1",
        async () => {
          const inputs = tokenizer("hello");
          const outputs = await model.generate({
            ...inputs,
            max_length: 10,
          });
          expect(outputs.tolist()).toEqual([[258n, 863n, 79n, 102n, 401n, 773n, 889n, 159n, 957n, 869n]]);
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
            [0n, 0n, 258n, 863n, 79n, 102n, 401n, 773n, 889n, 159n],
            [258n, 863n, 79n, 269n, 813n, 879n, 175n, 39n, 141n, 1000n],
          ]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await model?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });
  });

  describe("bloom", () => {
    describe("BloomForCausalLM", () => {
      const model_id = "hf-internal-testing/tiny-random-BloomForCausalLM";
      /** @type {BloomForCausalLM} */
      let model;
      /** @type {BloomTokenizer} */
      let tokenizer;
      beforeAll(async () => {
        model = await BloomForCausalLM.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
        tokenizer = await BloomTokenizer.from_pretrained(model_id);
      }, MAX_MODEL_LOAD_TIME);

      it(
        "batch_size=1",
        async () => {
          const inputs = tokenizer("hello");
          const outputs = await model.generate({
            ...inputs,
            max_length: 10,
          });
          expect(outputs.tolist()).toEqual([[198n, 803n, 82n, 82n, 82n, 82n, 82n, 82n, 82n, 82n]]);
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
            [3n, 3n, 198n, 803n, 82n, 82n, 82n, 82n, 82n, 82n],
            [198n, 803n, 82n, 209n, 753n, 753n, 753n, 753n, 753n, 753n],
          ]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await model?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });
  });

  describe("gpt_bigcode", () => {
    describe("GPTBigCodeForCausalLM", () => {
      const model_id = "hf-internal-testing/tiny-random-GPTBigCodeForCausalLM";
      /** @type {GPTBigCodeForCausalLM} */
      let model;
      /** @type {GPT2Tokenizer} */
      let tokenizer;
      beforeAll(async () => {
        model = await GPTBigCodeForCausalLM.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
        tokenizer = await GPT2Tokenizer.from_pretrained(model_id);
        tokenizer.padding_side = "left";
      }, MAX_MODEL_LOAD_TIME);

      it(
        "batch_size=1",
        async () => {
          const inputs = tokenizer("hello");
          const outputs = await model.generate({
            ...inputs,
            max_length: 10,
          });
          expect(outputs.tolist()).toEqual([[258n, 863n, 79n, 79n, 79n, 79n, 79n, 79n, 79n, 79n]]);
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
            [0n, 0n, 258n, 863n, 79n, 79n, 79n, 79n, 79n, 79n],
            [258n, 863n, 79n, 269n, 813n, 832n, 93n, 93n, 93n, 93n],
          ]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await model?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });
  });

  describe("gpt2", () => {
    describe("GPT2LMHeadModel", () => {
      const model_id = "hf-internal-testing/tiny-random-GPT2LMHeadModel";
      /** @type {GPT2LMHeadModel} */
      let model;
      /** @type {GPT2Tokenizer} */
      let tokenizer;
      beforeAll(async () => {
        model = await GPT2LMHeadModel.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
        tokenizer = await GPT2Tokenizer.from_pretrained(model_id);
        tokenizer.padding_side = "left";
      }, MAX_MODEL_LOAD_TIME);

      it(
        "batch_size=1",
        async () => {
          const inputs = tokenizer("hello");
          const outputs = await model.generate({
            ...inputs,
            max_length: 10,
          });
          expect(outputs.tolist()).toEqual([[258n, 863n, 79n, 79n, 79n, 79n, 79n, 79n, 79n, 243n]]);
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
            [0n, 0n, 258n, 863n, 79n, 79n, 79n, 79n, 79n, 79n],
            [258n, 863n, 79n, 269n, 813n, 813n, 813n, 813n, 813n, 813n],
          ]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await model?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });
  });

  describe("jais", () => {
    describe("JAISLMHeadModel", () => {
      const model_id = "onnx-community/tiny-random-jais";
      /** @type {JAISLMHeadModel} */
      let model;
      /** @type {PreTrainedTokenizer} */
      let tokenizer;
      beforeAll(async () => {
        model = await JAISLMHeadModel.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
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
          });
          expect(outputs.tolist()).toEqual([[55422n, 55422n, 55422n, 55422n, 55422n, 55422n, 55422n, 55422n, 55422n, 55422n]]);
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
            [0n, 55422n, 55422n, 55422n, 55422n, 55422n, 55422n, 55422n, 55422n, 55422n],
            [55422n, 2838n, 2838n, 2838n, 2838n, 2838n, 2838n, 2838n, 2838n, 2838n],
          ]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await model?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });
  });

  describe("mpt", () => {
    describe("MptForCausalLM", () => {
      const model_id = "hf-internal-testing/tiny-random-MptForCausalLM";
      /** @type {MptForCausalLM} */
      let model;
      /** @type {GPTNeoXTokenizer} */
      let tokenizer;
      beforeAll(async () => {
        model = await MptForCausalLM.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
        tokenizer = await GPTNeoXTokenizer.from_pretrained(model_id);
        tokenizer.padding_side = "left";
      }, MAX_MODEL_LOAD_TIME);

      it(
        "batch_size=1",
        async () => {
          const inputs = tokenizer("hello");
          const outputs = await model.generate({
            ...inputs,
            max_length: 10,
          });
          expect(outputs.tolist()).toEqual([[259n, 864n, 80n, 80n, 80n, 80n, 80n, 80n, 80n, 80n]]);
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
            [0n, 0n, 259n, 864n, 80n, 80n, 80n, 80n, 80n, 80n],
            [259n, 864n, 80n, 270n, 814n, 293n, 293n, 293n, 293n, 293n],
          ]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await model?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });
  });

  describe("codegen", () => {
    describe("CodeGenForCausalLM", () => {
      const model_id = "hf-internal-testing/tiny-random-CodeGenForCausalLM";
      /** @type {CodeGenForCausalLM} */
      let model;
      /** @type {CodeGenTokenizer} */
      let tokenizer;
      beforeAll(async () => {
        model = await CodeGenForCausalLM.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
        tokenizer = await CodeGenTokenizer.from_pretrained(model_id);
        tokenizer.padding_side = "left";
      }, MAX_MODEL_LOAD_TIME);

      it(
        "batch_size=1",
        async () => {
          const inputs = tokenizer("hello");
          const outputs = await model.generate({
            ...inputs,
            max_length: 10,
          });
          expect(outputs.tolist()).toEqual([[258n, 863n, 79n, 437n, 334n, 450n, 294n, 621n, 375n, 385n]]);
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
            [0n, 0n, 258n, 863n, 79n, 437n, 334n, 450n, 294n, 621n],
            [258n, 863n, 79n, 269n, 813n, 759n, 113n, 295n, 574n, 987n],
          ]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await model?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });
  });

  describe("mistral", () => {
    describe("MistralForCausalLM", () => {
      const model_id = "hf-internal-testing/tiny-random-MistralForCausalLM";
      /** @type {MistralForCausalLM} */
      let model;
      /** @type {LlamaTokenizer} */
      let tokenizer;
      beforeAll(async () => {
        model = await MistralForCausalLM.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
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
          expect(outputs.tolist()).toEqual([[1n, 6312n, 28709n, 24704n, 8732n, 1310n, 9808n, 13771n, 27309n, 4779n]]);
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
            [2n, 1n, 6312n, 28709n, 24704n, 8732n, 1310n, 9808n, 13771n, 27309n],
            [1n, 6312n, 28709n, 1526n, 8687n, 5690n, 1770n, 30811n, 12501n, 3325n],
          ]);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await model?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });
  });

  describe("patchtsmixer", () => {
    const dims = [64, 512, 7];
    const prod = dims.reduce((a, b) => a * b, 1);
    const past_values = new Tensor(
      "float32",
      Float32Array.from({ length: prod }, (_, i) => i / prod),
      dims,
    );

    describe("PatchTSMixerModel", () => {
      const model_id = "hf-internal-testing/tiny-random-PatchTSMixerModel";

      /** @type {PatchTSMixerModel} */
      let model;
      beforeAll(async () => {
        model = await PatchTSMixerModel.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
      }, MAX_MODEL_LOAD_TIME);

      it(
        "default",
        async () => {
          const { last_hidden_state } = await model({ past_values });

          const { num_input_channels, num_patches, d_model } = model.config;
          expect(last_hidden_state.dims).toEqual([dims[0], num_input_channels, num_patches, d_model]);
          expect(last_hidden_state.mean().item()).toBeCloseTo(0.03344963490962982, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await model?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });

    describe("PatchTSMixerForPrediction", () => {
      const model_id = "onnx-community/granite-timeseries-patchtsmixer";

      /** @type {PatchTSMixerForPrediction} */
      let model;
      beforeAll(async () => {
        model = await PatchTSMixerForPrediction.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
      }, MAX_MODEL_LOAD_TIME);

      it(
        "default",
        async () => {
          const { prediction_outputs } = await model({ past_values });

          const { prediction_length, num_input_channels } = model.config;
          expect(prediction_outputs.dims).toEqual([dims[0], prediction_length, num_input_channels]);
          expect(prediction_outputs.mean().item()).toBeCloseTo(0.5064773559570312, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await model?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });
  });

  describe("patchtst", () => {
    const dims = [64, 512, 7];
    const prod = dims.reduce((a, b) => a * b, 1);
    const past_values = new Tensor(
      "float32",
      Float32Array.from({ length: prod }, (_, i) => i / prod),
      dims,
    );

    describe("PatchTSTModel", () => {
      const model_id = "hf-internal-testing/tiny-random-PatchTSTModel";

      /** @type {PatchTSTModel} */
      let model;
      beforeAll(async () => {
        model = await PatchTSTModel.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
      }, MAX_MODEL_LOAD_TIME);

      it(
        "default",
        async () => {
          const { last_hidden_state } = await model({ past_values });

          const { num_input_channels, d_model } = model.config;
          expect(last_hidden_state.dims).toEqual([dims[0], num_input_channels, 43, d_model]);
          expect(last_hidden_state.mean().item()).toBeCloseTo(0.016672514379024506, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await model?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });

    describe("PatchTSTForPrediction", () => {
      const model_id = "onnx-community/granite-timeseries-patchtst";

      /** @type {PatchTSTForPrediction} */
      let model;
      beforeAll(async () => {
        model = await PatchTSTForPrediction.from_pretrained(model_id, {
          // TODO move to config
          ...DEFAULT_MODEL_OPTIONS,
        });
      }, MAX_MODEL_LOAD_TIME);

      it(
        "default",
        async () => {
          const { prediction_outputs } = await model({ past_values });

          const { prediction_length, num_input_channels } = model.config;
          expect(prediction_outputs.dims).toEqual([dims[0], prediction_length, num_input_channels]);
          expect(prediction_outputs.mean().item()).toBeCloseTo(0.506528377532959, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await model?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });
  });
});

describe("Tiny random pipelines", () => {
  describe("fill-mask", () => {
    const model_id = "hf-internal-testing/tiny-random-BertForMaskedLM";

    /** @type {FillMaskPipeline} */
    let pipe;
    beforeAll(async () => {
      pipe = await pipeline("fill-mask", model_id, {
        // TODO move to config
        ...DEFAULT_MODEL_OPTIONS,
      });
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
      pipe = await pipeline("text-classification", model_id, {
        // TODO move to config
        ...DEFAULT_MODEL_OPTIONS,
      });
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
      pipe = await pipeline("token-classification", model_id, {
        // TODO move to config
        ...DEFAULT_MODEL_OPTIONS,
      });
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
      pipe = await pipeline("question-answering", model_id, {
        // TODO move to config
        ...DEFAULT_MODEL_OPTIONS,
      });
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
      pipe = await pipeline("image-classification", model_id, {
        // TODO move to config
        ...DEFAULT_MODEL_OPTIONS,
      });
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
      pipe = await pipeline("zero-shot-image-classification", model_id, {
        // TODO move to config
        ...DEFAULT_MODEL_OPTIONS,
      });
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
      pipe = await pipeline("audio-classification", model_id, {
        // TODO move to config
        ...DEFAULT_MODEL_OPTIONS,
      });
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
      pipe = await pipeline("text-generation", model_id, {
        // TODO move to config
        ...DEFAULT_MODEL_OPTIONS,
      });
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
      pipe = await pipeline("translation", model_id, {
        // TODO move to config
        ...DEFAULT_MODEL_OPTIONS,
      });
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
      pipe = await pipeline("object-detection", model_id, {
        // TODO move to config
        ...DEFAULT_MODEL_OPTIONS,
      });
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
      pipe = await pipeline("document-question-answering", model_id, {
        // TODO move to config
        ...DEFAULT_MODEL_OPTIONS,
      });
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

describe("PKV caching", () => {
  describe("LlamaForCausalLM", () => {
    const model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM";
    /** @type {LlamaForCausalLM} */
    let model;
    /** @type {LlamaTokenizer} */
    let tokenizer;
    beforeAll(async () => {
      model = await LlamaForCausalLM.from_pretrained(model_id, {
        // TODO move to config
        ...DEFAULT_MODEL_OPTIONS,
      });
      tokenizer = await LlamaTokenizer.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "batch_size=1",
      async () => {
        const inputs = tokenizer("1");

        // Generate first sequence w/o PKV
        // NOTE: `return_dict_in_generate=true` is required to get PKV
        const { past_key_values, sequences } = await model.generate({
          ...inputs,
          max_new_tokens: 5,
          do_sample: false,
          return_dict_in_generate: true,
        });

        // Update output with new text
        const decoded = tokenizer.batch_decode(sequences, {
          skip_special_tokens: false,
        })[0];
        const new_inputs = tokenizer(decoded + "2", {
          add_special_tokens: false,
        });

        // Run w/o PKV
        const generated_ids = await model.generate({
          ...new_inputs,
          max_new_tokens: 3,
          do_sample: false,
        });

        // Run w/ PKV
        const generated_ids_pkv = await model.generate({
          ...new_inputs,
          past_key_values,
          max_new_tokens: 3,
          do_sample: false,
        });

        const target = [[1n, 259n, 29896n, 24959n, 22063n, 17192n, 12189n, 22468n, 29906n, 3399n, 24823n, 26470n]];

        expect(generated_ids.tolist()).toEqual(target);
        expect(generated_ids_pkv.tolist()).toEqual(target);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("LlavaForConditionalGeneration", () => {
    const model_id = "Xenova/tiny-random-LlavaForConditionalGeneration";
    /** @type {LlavaForConditionalGeneration} */
    let model;
    /** @type {PreTrainedTokenizer} */
    let tokenizer;
    /** @type {Processor} */
    let processor;
    beforeAll(async () => {
      model = await LlavaForConditionalGeneration.from_pretrained(model_id, {
        // TODO move to config
        ...DEFAULT_MODEL_OPTIONS,
      });
      tokenizer = await AutoTokenizer.from_pretrained(model_id);
      processor = await AutoProcessor.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "batch_size=1",
      async () => {
        const text_inputs = tokenizer("<image>hello");

        // Empty white image
        const dims = [224, 224, 3];
        const image = new RawImage(new Uint8ClampedArray(dims[0] * dims[1] * dims[2]).fill(255), ...dims);
        const vision_inputs = await processor(image);

        // Generate first sequence w/o PKV
        // NOTE: `return_dict_in_generate=true` is required to get PKV
        const { past_key_values, sequences } = await model.generate({
          ...text_inputs,
          ...vision_inputs,
          max_new_tokens: 5,
          do_sample: false,
          return_dict_in_generate: true,
        });

        // Update output with new text
        const decoded = tokenizer.batch_decode(sequences).map((x) => x + "new");
        const new_inputs = tokenizer(decoded, {
          add_special_tokens: false,
        });

        // Run w/o PKV
        const generated_ids = await model.generate({
          ...new_inputs,
          ...vision_inputs,
          max_new_tokens: 3,
          do_sample: false,
        });

        // Run w/ PKV
        const generated_ids_pkv = await model.generate({
          ...new_inputs,
          past_key_values,
          max_new_tokens: 3,
          do_sample: false,
        });

        const target = [[1n, 32000n, 29871n, 23927n, 359n, 1519n, 568n, 5769n, 1330n, 21544n, 11568n, 1482n, 7258n, 1250n, 16117n]];
        expect(generated_ids.tolist()).toEqual(target);
        expect(generated_ids_pkv.tolist()).toEqual(target);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
});
