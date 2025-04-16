import { pipeline, cos_sim } from "../src/transformers.js";
import { init, MAX_TEST_EXECUTION_TIME } from "./init.js";
import { collect_and_execute_pipeline_tests, compare, compareString, loadAudio } from "./test_utils.js";

// Initialise the testing environment
init();
await collect_and_execute_pipeline_tests("Pipelines");

// NOTE:
// Due to a memory leak in Jest, we cannot have multiple tests for a single model.
// This is due to how model construction and destruction occurs, in `beforeAll` and `afterAll`, respectively.
// As a result, each test is responsible for exactly one model, but we run multiple inputs through it.
// By encapsulating model construction and destruction in a single `it` block, we avoid these memory issues.
xdescribe("Pipelines (ignored)", () => {
  describe("Text classification", () => {
    // List all models which will be tested
    const models = ["Xenova/distilbert-base-uncased-finetuned-sst-2-english", "Xenova/toxic-bert"];

    // single_label_classification
    it(
      models[0],
      async () => {
        let classifier = await pipeline("text-classification", models[0]);
        let texts = ["This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three.", "I hated the movie"];

        // single
        {
          let outputs = await classifier("I hated the movie");
          let expected = [{ label: "NEGATIVE", score: 0.9996212720870972 }];
          compare(outputs, expected);
        }

        // single + topk
        {
          let outputs = await classifier("I hated the movie", {
            topk: 2,
          });
          let expected = [
            { label: "NEGATIVE", score: 0.9996212720870972 },
            { label: "POSITIVE", score: 0.0003787268069572747 },
          ];
          compare(outputs, expected);
        }

        // batched
        {
          let outputs = await classifier(texts);

          let expected = [
            { label: "POSITIVE", score: 0.9993746876716614 },
            { label: "NEGATIVE", score: 0.9996694326400757 },
          ];

          compare(outputs, expected);
        }

        // batched + topk
        {
          let outputs = await classifier(texts, {
            topk: 2,
          });

          let expected = [
            [
              { label: "POSITIVE", score: 0.9993746876716614 },
              { label: "NEGATIVE", score: 0.0006253048195503652 },
            ],
            [
              { label: "NEGATIVE", score: 0.9996694326400757 },
              { label: "POSITIVE", score: 0.00033057318069040775 },
            ],
          ];

          compare(outputs, expected);
        }

        await classifier.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );

    // multi_label_classification
    it(
      models[1],
      async () => {
        let classifier = await pipeline("text-classification", models[1]);
        let texts = [
          "I like you. I love you", // low scores
          "I hate you.", // high scores
        ];

        // single
        {
          let outputs = await classifier(texts);
          let expected = [
            { label: "toxic", score: 0.0007729064091108739 },
            { label: "toxic", score: 0.9475088119506836 },
          ];
          compare(outputs, expected);
        }
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });

  describe("Token classification", () => {
    // List all models which will be tested
    const models = ["Xenova/bert-base-multilingual-cased-ner-hrl"];

    it(
      models[0],
      async () => {
        let classifier = await pipeline("token-classification", models[0]);
        let texts = ["The Golden State Warriors are an American professional basketball team based in San Francisco.", "My name is Sarah and I live in London."];

        // single
        {
          let outputs = await classifier(texts[0]);

          let expected = [
            { entity: "B-ORG", score: 0.9998535513877869, index: 2, word: "Golden", start: null, end: null },
            { entity: "I-ORG", score: 0.9998612999916077, index: 3, word: "State", start: null, end: null },
            { entity: "I-ORG", score: 0.999866247177124, index: 4, word: "Warriors", start: null, end: null },
            { entity: "B-LOC", score: 0.9997050166130066, index: 13, word: "San", start: null, end: null },
            { entity: "I-LOC", score: 0.9987282156944275, index: 14, word: "Francisco", start: null, end: null },
          ];

          compare(outputs, expected, 0.05);
        }

        // batched
        {
          let outputs = await classifier(texts);

          let expected = [
            [
              { entity: "B-ORG", score: 0.9998375773429871, index: 2, word: "Golden", start: null, end: null },
              { entity: "I-ORG", score: 0.9998642206192017, index: 3, word: "State", start: null, end: null },
              { entity: "I-ORG", score: 0.9998642802238464, index: 4, word: "Warriors", start: null, end: null },
              { entity: "B-LOC", score: 0.9996914863586426, index: 13, word: "San", start: null, end: null },
              { entity: "I-LOC", score: 0.9989780783653259, index: 14, word: "Francisco", start: null, end: null },
            ],
            [
              { entity: "B-PER", score: 0.997977614402771, index: 4, word: "Sarah", start: null, end: null },
              { entity: "B-LOC", score: 0.9996902346611023, index: 9, word: "London", start: null, end: null },
            ],
          ];

          compare(outputs, expected, 0.05);
        }

        await classifier.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });

  describe("Zero-shot classification", () => {
    // List all models which will be tested
    const models = ["Xenova/bart-large-mnli"];

    it(
      models[0],
      async () => {
        let classifier = await pipeline("zero-shot-classification", models[0]);

        let sequences_to_classify = ["one day I will see the world", "I love making pizza"];
        let candidate_labels = ["travel", "cooking", "dancing"];

        // single
        {
          let outputs = await classifier(sequences_to_classify[0], candidate_labels);
          let expected = {
            sequence: "one day I will see the world",
            labels: ["travel", "dancing", "cooking"],
            scores: [0.4261703487477968, 0.2903585771517135, 0.28347107410048983],
          };

          compare(outputs, expected, 0.2);
        }

        // batched
        {
          let outputs = await classifier(sequences_to_classify, candidate_labels);
          let expected = [
            {
              sequence: "one day I will see the world",
              labels: ["travel", "dancing", "cooking"],
              scores: [0.4261703487477968, 0.2903585771517135, 0.28347107410048983],
            },
            {
              sequence: "I love making pizza",
              labels: ["cooking", "travel", "dancing"],
              scores: [0.4660367922118968, 0.2756005926506238, 0.2583626151374795],
            },
          ];

          compare(outputs, expected, 0.2);
        }

        // batched + multilabel
        {
          let outputs = await classifier(sequences_to_classify, candidate_labels, {
            multi_label: true,
          });
          let expected = [
            {
              sequence: "one day I will see the world",
              labels: ["travel", "dancing", "cooking"],
              scores: [0.7108286792234982, 0.5763787804099745, 0.44303326070949994],
            },
            {
              sequence: "I love making pizza",
              labels: ["cooking", "travel", "dancing"],
              scores: [0.8527619536354446, 0.7899589317978243, 0.5838912691496106],
            },
          ];

          compare(outputs, expected);
        }

        await classifier.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });

  describe("Masked language modelling", () => {
    // List all models which will be tested
    const models = ["Xenova/bert-base-uncased"];

    it(
      models[0],
      async () => {
        let unmasker = await pipeline("fill-mask", models[0]);
        let texts = ["Once upon a [MASK].", "[MASK] is the capital of England."];

        // single
        {
          let outputs = await unmasker(texts[0]);
          let expected = [
            {
              score: 0.9405396580696106,
              token: 2051,
              token_str: "time",
              sequence: "once upon a time.",
            },
            {
              score: 0.01182964164763689,
              token: 13342,
              token_str: "mattress",
              sequence: "once upon a mattress.",
            },
            {
              score: 0.0017291896510869265,
              token: 6480,
              token_str: "lifetime",
              sequence: "once upon a lifetime.",
            },
            {
              score: 0.0010079898638650775,
              token: 2504,
              token_str: "level",
              sequence: "once upon a level.",
            },
            {
              score: 0.0009655007743276656,
              token: 2154,
              token_str: "day",
              sequence: "once upon a day.",
            },
          ];
          compare(outputs, expected);
        }

        // batched
        {
          let outputs = await unmasker(texts);

          let expected = [
            [
              {
                score: 0.9900539517402649,
                token: 2051,
                token_str: "time",
                sequence: "once upon a time.",
              },
              {
                score: 0.0012258145725354552,
                token: 13342,
                token_str: "mattress",
                sequence: "once upon a mattress.",
              },
              {
                score: 0.0002977887343149632,
                token: 2096,
                token_str: "while",
                sequence: "once upon a while.",
              },
              {
                score: 0.0001899998023873195,
                token: 6480,
                token_str: "lifetime",
                sequence: "once upon a lifetime.",
              },
              {
                score: 0.00017618606216274202,
                token: 2558,
                token_str: "period",
                sequence: "once upon a period.",
              },
            ],
            [
              {
                score: 0.2863538861274719,
                token: 2414,
                token_str: "london",
                sequence: "london is the capital of england.",
              },
              {
                score: 0.0607745461165905,
                token: 2009,
                token_str: "it",
                sequence: "it is the capital of england.",
              },
              {
                score: 0.037455108016729355,
                token: 6484,
                token_str: "birmingham",
                sequence: "birmingham is the capital of england.",
              },
              {
                score: 0.029375044628977776,
                token: 5087,
                token_str: "manchester",
                sequence: "manchester is the capital of england.",
              },
              {
                score: 0.0292277242988348,
                token: 7067,
                token_str: "bristol",
                sequence: "bristol is the capital of england.",
              },
            ],
          ];

          compare(outputs, expected);
        }

        await unmasker.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });

  describe("Question answering", () => {
    let question = "Who was Jim Henson?";
    let context = "Jim Henson was a nice puppet.";

    // List all models which will be tested
    const models = ["Xenova/distilbert-base-uncased-distilled-squad"];

    it(
      models[0],
      async () => {
        let answerer = await pipeline("question-answering", models[0]);

        // single
        {
          let outputs = await answerer(question, context);
          let expected = { answer: "a nice puppet", score: 0.5664517526948352 };

          compare(outputs, expected, 0.2);
        }

        // single + topk
        {
          let outputs = await answerer(question, context, {
            topk: 3,
          });
          let expected = [
            { answer: "a nice puppet", score: 0.5664517526948352 },
            { answer: "nice puppet", score: 0.1698902336448853 },
            { answer: "puppet", score: 0.14046057793125577 },
          ];

          compare(outputs, expected, 0.2);
        }
        await answerer.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });

  describe("Summarization", () => {
    // List all models which will be tested
    const models = ["Xenova/distilbart-cnn-6-6", "Xenova/bart-large-cnn"];

    let texts = [`The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.`, `The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish: Selva Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch: Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which 5,500,000 square kilometres (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia, Guyana, Suriname and French Guiana. States or departments in four nations contain "Amazonas" in their names. The Amazon represents over half of the planet's remaining rainforests, and comprises the largest and most biodiverse tract of tropical rainforest in the world, with an estimated 390 billion individual trees divided into 16,000 species.`];

    it(
      models[0],
      async () => {
        let summarizer = await pipeline("summarization", models[0]);

        // batched
        {
          let summary = await summarizer(texts, {
            top_k: 0,
            do_sample: false,
          });
          expect(summary).toHaveLength(2);
          expect(summary[0].summary_text.length).toBeGreaterThan(50);
          expect(summary[1].summary_text.length).toBeGreaterThan(50);
        }
        await summarizer.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      models[1],
      async () => {
        let summarizer = await pipeline("summarization", models[1]);

        // batched + `forced_bos_token_id`
        {
          let summary = await summarizer(texts[0], {
            top_k: 0,
            do_sample: false,
          });
          expect(summary).toHaveLength(1);
          expect(summary[0].summary_text.length).toBeGreaterThan(50);
        }

        await summarizer.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });

  describe("Translation", () => {
    // List all models which will be tested
    const models = [
      "Xenova/t5-small",

      // Multilingual model
      "Xenova/nllb-200-distilled-600M",
    ];

    it(
      models[0],
      async () => {
        let translator = await pipeline("translation_en_to_de", models[0]);
        let texts = ["Hello, how are you?", "My name is Maria."];

        // single
        {
          let translation = await translator(texts[0], {
            top_k: 0,
            do_sample: false,
          });

          let expected = [{ translation_text: "Hallo, wie sind Sie?" }];

          compare(translation, expected);
        }

        // batched
        {
          let output = await translator(texts, {
            top_k: 0,
            do_sample: false,
          });

          let expected = [{ translation_text: "Hallo, wie sind Sie?" }, { translation_text: "Mein Name ist Maria." }];

          compare(output, expected);
        }

        await translator.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      models[1],
      async () => {
        let translator = await pipeline("translation", models[1]);
        let texts = ["Hello world!", "I like to walk my dog."];

        // single
        {
          let translation = await translator(texts[0], {
            src_lang: "eng_Latn",
            tgt_lang: "arb_Arab",
          });

          let expected = [{ translation_text: "مرحباً، يا عالم!" }];

          compare(translation, expected);
        }

        // single + back-translation
        {
          let translation1 = await translator(texts[1], {
            // src_lang: 'eng_Latn',
            tgt_lang: "ell_Grek",
          });
          let translation2 = await translator(translation1[0].translation_text, {
            src_lang: "ell_Grek",
            tgt_lang: "eng_Latn",
          });

          let expected = [{ translation_text: "Μου αρέσει να περπατάω το σκυλί μου." }];

          compare(translation1, expected);

          let expectedBack = [{ translation_text: texts[1] }];
          compare(translation2, expectedBack);
        }

        await translator.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });

  describe("Text-to-text generation", () => {
    // List all models which will be tested
    const models = ["Xenova/flan-t5-small", "Xenova/flan-t5-base"];

    it(
      models[0],
      async () => {
        let generator = await pipeline("text2text-generation", models[0]);
        let text = "Premise:  At my age you will probably have learnt one lesson. " + "Hypothesis:  It's not certain how many lessons you'll learn by your thirties. " + "Does the premise entail the hypothesis?";

        {
          let outputs = await generator(text, {
            top_k: 0,
            do_sample: false,
          });
          expect(outputs).toHaveLength(1);
          expect(outputs[0].generated_text.length).toBeGreaterThan(1);
        }

        await generator.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      models[1],
      async () => {
        let generator = await pipeline("text2text-generation", models[1]);
        let text = `
            Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can
            has 3 tennis balls. How many tennis balls does he have now?
            A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls.
            5 + 6 = 11. The answer is 11.

            Q: A juggler can juggle 16 balls. Half of the balls are golf balls, and half
            of the golf balls are blue. How many blue golf balls are there?`;

        // single
        {
          let outputs = await generator(text, {
            top_k: 0,
            do_sample: false,
          });
          expect(outputs).toHaveLength(1);
          expect(outputs[0].generated_text.length).toBeGreaterThan(10);
        }
        await generator.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });

  describe("Text generation", () => {
    // List all models which will be tested
    const models = ["Xenova/distilgpt2", "Xenova/codegen-350M-mono"];

    it(
      models[0],
      async () => {
        let generator = await pipeline("text-generation", models[0]);
        let texts = ["Once upon a time, there was a", "I enjoy walking with my cute dog"];

        // single
        {
          let output = await generator(texts[0], {
            max_new_tokens: 10,
            top_k: 0,
            do_sample: false,
          });
          expect(output).toHaveLength(1);
          expect(output[0].generated_text.length).toBeGreaterThan(texts[0].length);
        }

        // single + `num_beams` + `num_return_sequences`
        {
          let output = await generator(texts[0], {
            max_new_tokens: 10,
            num_beams: 2,
            num_return_sequences: 2,
            top_k: 0,
            do_sample: false,
          });
          expect(output).toHaveLength(2);
          expect(output[0].generated_text.length).toBeGreaterThan(texts[0].length);
          expect(output[1].generated_text.length).toBeGreaterThan(texts[0].length);
        }

        // batched + `num_beams` + `num_return_sequences`
        {
          let output = await generator(texts, {
            max_new_tokens: 10,
            num_beams: 2,
            num_return_sequences: 2,
            top_k: 0,
            do_sample: false,
          });
          expect(output).toHaveLength(2);
          expect(output[0]).toHaveLength(2);
          expect(output[0][0].generated_text.length).toBeGreaterThan(texts[0].length);
          expect(output[0][1].generated_text.length).toBeGreaterThan(texts[0].length);
          expect(output[1]).toHaveLength(2);
          expect(output[1][0].generated_text.length).toBeGreaterThan(texts[1].length);
          expect(output[1][1].generated_text.length).toBeGreaterThan(texts[1].length);
        }

        await generator.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      models[1],
      async () => {
        let generator = await pipeline("text-generation", models[1]);
        let code = "def fib(n):";

        // single + `added_tokens`
        {
          let output = await generator(code, {
            max_new_tokens: 45,
            top_k: 0,
            do_sample: false,
          });
          expect(output).toHaveLength(1);
          expect(output[0].generated_text.length).toBeGreaterThan(code.length);
        }
        await generator.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });

  describe("Feature extraction", () => {
    // List all models which will be tested
    const models = ["Xenova/all-MiniLM-L6-v2"];

    it(
      models[0],
      async () => {
        let extractor = await pipeline("feature-extraction", models[0]);

        // Provide sentences
        let sentences = ["This framework generates embeddings for each input sentence", "Sentences are passed as a list of string.", "The quick brown fox jumps over the lazy dog."];

        // Without pooling or normalization
        {
          let output = await extractor(sentences);
          expect(output.dims).toHaveLength(3);
        }

        // With pooling and normalization + compare features
        {
          let output = await extractor(sentences, { pooling: "mean", normalize: true });
          expect(output.dims).toHaveLength(2);

          // Convert Tensor to JS list
          output = output.tolist();

          let pairwiseScores = [
            [output[0], output[1]],
            [output[0], output[2]],
            [output[1], output[2]],
          ].map((x) => cos_sim(...x));

          let expected = [0.502872309810269, 0.11088411026413121, 0.09602621986931259];
          compare(pairwiseScores, expected);
        }
        await extractor.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });

  describe("Speech-to-text generation", () => {
    // List all models which will be tested
    const models = [
      // whisper
      "Xenova/whisper-tiny.en", // English-only
      "Xenova/whisper-small", // Multilingual
      ["Xenova/whisper-tiny.en", "output_attentions"], // English-only + `output_attentions`
      ["Xenova/whisper-small", "output_attentions"], // Multilingual + `output_attentions`

      // wav2vec2
      "jonatasgrosman/wav2vec2-large-xlsr-53-english",
    ];

    it(
      models[0],
      async () => {
        let transcriber = await pipeline("automatic-speech-recognition", models[0]);

        let url = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav";
        let audioData = await loadAudio(url);

        {
          // Transcribe English
          let output = await transcriber(audioData);
          expect(output.text.length).toBeGreaterThan(50);
          const expected = " And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.";
          compareString(expected, output.text);
        }

        {
          // Transcribe English w/ timestamps.
          let output = await transcriber(audioData, { return_timestamps: true });
          expect(output.text.length).toBeGreaterThan(50);
          expect(output.chunks.length).toBeGreaterThan(0);
          // {
          //   text: " And so my fellow Americans ask not what your country can do for you, ask what you can do for your country."
          //   chunks: [
          //     { timestamp: [0, 8],  text: " And so my fellow Americans ask not what your country can do for you" }
          //     { timestamp: [8, 11], text: " ask what you can do for your country." }
          //   ]
          // }
        }
        await transcriber.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      models[1],
      async () => {
        let transcriber = await pipeline("automatic-speech-recognition", models[1]);

        let url = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/french-audio.wav";
        let audioData = await loadAudio(url);

        {
          // Transcribe French
          let output = await transcriber(audioData, { language: "french", task: "transcribe" });
          expect(output.text.length).toBeGreaterThan(20);
          const expected = " J'adore, j'aime, je n'aime pas, je déteste.";
          compareString(expected, output.text);
        }

        {
          // Translate French to English.
          let output = await transcriber(audioData, { language: "french", task: "translate" });
          expect(output.text.length).toBeGreaterThan(20);
          const expected = " I love, I like, I don't like, I hate.";
          compareString(expected, output.text);
        }
        await transcriber.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );
    
    it(
      `${models[1]}-language-detect`,
      async () => {
        let transcriber = await pipeline("automatic-speech-recognition", models[1]);
        let url = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/japanese-audio.wav";
        let audioData = await loadAudio(url);
        {
          // Transcribe Japanese by autodetecting language
          // Note: this sample needs to be hard enough for Whisper not to be able to transcribe it properly
          //  with the fallback 'en' language set!
          let output = await transcriber(audioData, { language: null, task: "transcribe" });
          expect(output.text.length).toBeGreaterThan(20);
          const expected = "モリナガの美味しい牛乳は濃い青色に牛乳瓶を払ったゼザインのパック牛乳である。";
          compareString(expected, output.text, 0.8);
        }
        await transcriber.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );
    
    it(
      models[2].join(" + "),
      async () => {
        let transcriber = await pipeline("automatic-speech-recognition", m(models[2][0]), {
          revision: models[2][1],
          quantized: false,
        });

        let url = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav";
        let audioData = await loadAudio(url);

        {
          // Transcribe English w/ word-level timestamps.
          let output = await transcriber(audioData, { return_timestamps: "word" });
          const target = {
            text: " And so my fellow Americans ask not what your country can do for you ask what you can do for your country.",
            chunks: [
              { text: " And", timestamp: [0, 0.78] },
              { text: " so", timestamp: [0.78, 1.06] },
              { text: " my", timestamp: [1.06, 1.46] },
              { text: " fellow", timestamp: [1.46, 1.76] },
              { text: " Americans", timestamp: [1.76, 2.22] },
              { text: " ask", timestamp: [2.22, 3.88] },
              { text: " not", timestamp: [3.88, 4.52] },
              { text: " what", timestamp: [4.52, 5.68] },
              { text: " your", timestamp: [5.68, 6] },
              { text: " country", timestamp: [6, 6.36] },
              { text: " can", timestamp: [6.36, 6.76] },
              { text: " do", timestamp: [6.76, 7.02] },
              { text: " for", timestamp: [7.02, 7.24] },
              { text: " you", timestamp: [7.24, 8.02] },
              { text: " ask", timestamp: [8.28, 8.66] },
              { text: " what", timestamp: [8.66, 8.94] },
              { text: " you", timestamp: [8.94, 9.28] },
              { text: " can", timestamp: [9.28, 9.5] },
              { text: " do", timestamp: [9.5, 9.72] },
              { text: " for", timestamp: [9.72, 9.92] },
              { text: " your", timestamp: [9.92, 10.22] },
              { text: " country.", timestamp: [10.22, 13.36] },
            ],
          };

          compare(output, target);
        }

        await transcriber.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      models[3].join(" + "),
      async () => {
        let transcriber = await pipeline("automatic-speech-recognition", m(models[3][0]), {
          revision: models[3][1],
        });

        let url = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/japanese-audio.wav";
        let audioData = await loadAudio(url);

        {
          // Transcribe Japanese w/ word-level timestamps.
          let output = await transcriber(audioData, { return_timestamps: "word", language: "japanese", task: "transcribe" });
          const target = {
            text: "モリナガの美味しい牛乳は濃い青色に牛乳瓶を払ったゼザインのパック牛乳である。",
            chunks: [
              { text: "モ", timestamp: [0, 0.56] },
              { text: "リ", timestamp: [0.56, 0.64] },
              { text: "ナ", timestamp: [0.64, 0.8] },
              { text: "ガ", timestamp: [0.8, 0.88] },
              { text: "の", timestamp: [0.88, 1.04] },
              { text: "美味", timestamp: [1.04, 1.22] },
              { text: "しい", timestamp: [1.22, 1.46] },
              { text: "牛", timestamp: [1.46, 1.76] },
              { text: "乳", timestamp: [1.76, 1.94] },
              { text: "は", timestamp: [1.94, 2.14] },
              { text: "濃", timestamp: [2.14, 2.34] },
              { text: "い", timestamp: [2.34, 2.48] },
              { text: "青", timestamp: [2.48, 2.62] },
              { text: "色", timestamp: [2.62, 2.84] },
              { text: "に", timestamp: [2.84, 3] },
              { text: "牛", timestamp: [3, 3.22] },
              { text: "乳", timestamp: [3.22, 3.42] },
              { text: "瓶", timestamp: [3.42, 3.58] },
              { text: "を", timestamp: [3.58, 3.82] },
              { text: "払", timestamp: [3.82, 4] },
              { text: "った", timestamp: [4, 4.32] },
              { text: "ゼ", timestamp: [4.32, 4.56] },
              { text: "ザ", timestamp: [4.56, 4.6] },
              { text: "イ", timestamp: [4.6, 4.74] },
              { text: "ン", timestamp: [4.74, 4.8] },
              { text: "の", timestamp: [4.8, 4.94] },
              { text: "パ", timestamp: [4.94, 5.12] },
              { text: "ック", timestamp: [5.12, 5.26] },
              { text: "牛", timestamp: [5.26, 5.52] },
              { text: "乳", timestamp: [5.52, 5.72] },
              { text: "で", timestamp: [5.72, 5.86] },
              { text: "ある。", timestamp: [5.86, 6.62] },
            ],
          };

          compare(output, target);
        }

        await transcriber.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      models[4],
      async () => {
        let transcriber = await pipeline("automatic-speech-recognition", m(models[4]));

        let url = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav";
        let audioData = await loadAudio(url);

        {
          // Transcribe
          let output = await transcriber(audioData);
          expect(output.text.length).toBeGreaterThan(50);
          // { text: "and so my fellow america ask not what your country can do for you ask what you can do for your country" }
        }

        await transcriber.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });

  describe("Text-to-speech generation", () => {
    // List all models which will be tested
    const models = ["Xenova/speecht5_tts", "Xenova/mms-tts-fra"];

    it(
      models[0],
      async () => {
        let synthesizer = await pipeline("text-to-speech", models[0], {
          // NOTE: Although the quantized version produces incoherent results,
          // it it is okay to use for testing.
          // quantized: false,
        });

        let speaker_embeddings = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/speaker_embeddings.bin";

        {
          // Generate English speech
          let output = await synthesizer("Hello, my dog is cute", { speaker_embeddings });
          expect(output.audio.length).toBeGreaterThan(0);
          expect(output.sampling_rate).toEqual(16000);
        }

        await synthesizer.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      models[1],
      async () => {
        let synthesizer = await pipeline("text-to-speech", models[1]);

        {
          // Generate French speech
          let output = await synthesizer("Bonjour");
          expect(output.audio.length).toBeGreaterThan(0);
          expect(output.sampling_rate).toEqual(16000);
        }

        await synthesizer.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });

  describe("Audio classification", () => {
    // List all models which will be tested
    const models = ["Xenova/wav2vec2-large-xlsr-53-gender-recognition-librispeech"];

    it(
      models[0],
      async () => {
        let classifier = await pipeline("audio-classification", models[0]);

        let url = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav";
        let audioData = await loadAudio(url);

        {
          // Classify audio
          let outputs = await classifier(audioData);

          let expected = [
            { score: 0.997512936592102, label: "male" },
            { score: 0.0024870133493095636, label: "female" },
          ];
          compare(outputs, expected);
        }

        await classifier.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });

  describe("Image-to-text", () => {
    // List all models which will be tested
    const models = ["Xenova/vit-gpt2-image-captioning"];

    it(
      models[0],
      async () => {
        let captioner = await pipeline("image-to-text", models[0]);

        let url = "https://huggingface.co/datasets/mishig/sample_images/resolve/main/savanna.jpg";
        let urls = ["https://huggingface.co/datasets/mishig/sample_images/resolve/main/football-match.jpg", "https://huggingface.co/datasets/mishig/sample_images/resolve/main/airport.jpg"];

        // single
        {
          let output = await captioner(url, {
            top_k: 0,
            do_sample: false,
          });
          // let expected = [
          //     { "generated_text": "a herd of giraffes and zebras grazing in a field" }
          // ]

          expect(output).toHaveLength(1);
          expect(output[0].generated_text.length).toBeGreaterThan(10);
        }

        // single + generation options
        {
          let output = await captioner(url, {
            max_new_tokens: 20,
            num_beams: 2,
            num_return_sequences: 2,
            top_k: 0,
            do_sample: false,
          });
          // let expected = [
          //     { "generated_text": "a herd of giraffes and zebras grazing in a field" },
          //     { "generated_text": "a herd of giraffes and zebras in a grassy field" }
          // ]

          expect(output).toHaveLength(2);
          expect(output[0].generated_text.length).toBeGreaterThan(10);
          expect(output[1].generated_text.length).toBeGreaterThan(10);
        }

        // batched
        {
          let output = await captioner(urls, {
            top_k: 0,
            do_sample: false,
          });
          // let expected = [
          //     [{ "generated_text": "two men are kicking a soccer ball in a soccer game" }],
          //     [{ "generated_text": "a plane on the tarmac with a passenger bus" }]
          // ]

          expect(output).toHaveLength(2);
          expect(output[0]).toHaveLength(1);
          expect(output[0][0].generated_text.length).toBeGreaterThan(10);
          expect(output[1]).toHaveLength(1);
          expect(output[1][0].generated_text.length).toBeGreaterThan(10);
        }

        // batched + generation options
        {
          let output = await captioner(urls, {
            max_new_tokens: 20,
            num_beams: 2,
            num_return_sequences: 2,
            top_k: 0,
            do_sample: false,
          });
          // let expected = [
          //     [
          //         { "generated_text": "two men are kicking a soccer ball on a field" },
          //         { "generated_text": "two men are kicking a soccer ball in a soccer game" }
          //     ], [
          //         { "generated_text": "a plane on a tarmac with a group of buses" },
          //         { "generated_text": "a plane on a tarmac with a group of people on the ground" }
          //     ]
          // ];

          expect(output).toHaveLength(2);
          expect(output[0]).toHaveLength(2);
          expect(output[0][0].generated_text.length).toBeGreaterThan(10);
          expect(output[0][1].generated_text.length).toBeGreaterThan(10);
          expect(output[1]).toHaveLength(2);
          expect(output[1][0].generated_text.length).toBeGreaterThan(10);
          expect(output[1][1].generated_text.length).toBeGreaterThan(10);
        }
        await captioner.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });

  describe("Image classification", () => {
    // List all models which will be tested
    const models = ["Xenova/vit-base-patch16-224"];

    it(
      models[0],
      async () => {
        let classifier = await pipeline("image-classification", models[0]);

        let url = "https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg";
        let urls = ["https://huggingface.co/datasets/mishig/sample_images/resolve/main/palace.jpg", "https://huggingface.co/datasets/mishig/sample_images/resolve/main/teapot.jpg"];

        // single
        {
          let outputs = await classifier(url);

          let expected = [{ label: "tiger, Panthera tigris", score: 0.607988178730011 }];

          compare(outputs, expected, 0.2);
        }

        // single + topk
        {
          let outputs = await classifier(url, {
            topk: 2,
          });

          let expected = [
            { label: "tiger, Panthera tigris", score: 0.607988178730011 },
            { label: "tiger cat", score: 0.3877776563167572 },
          ];

          compare(outputs, expected, 0.2);
        }

        // batched
        {
          let outputs = await classifier(urls);

          let expected = [
            { label: "palace", score: 0.9986862540245056 },
            { label: "teapot", score: 0.987880527973175 },
          ];

          compare(outputs, expected);
        }

        // batched + topk
        {
          let outputs = await classifier(urls, {
            topk: 2,
          });

          let expected = [
            [
              { label: "palace", score: 0.9986862540245056 },
              { label: "castle", score: 0.00037879671435803175 },
            ],
            [
              { label: "teapot", score: 0.987880527973175 },
              { label: "coffeepot", score: 0.006591461598873138 },
            ],
          ];

          compare(outputs, expected);
        }

        await classifier.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });

  describe("Image segmentation", () => {
    // List all models which will be tested
    const models = ["Xenova/detr-resnet-50-panoptic", "Xenova/segformer_b2_clothes"];

    it(
      models[0],
      async () => {
        let segmenter = await pipeline("image-segmentation", models[0], {
          // Quantized version of model produces incorrect results
          quantized: false,
        });
        let img = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png";

        // single
        {
          let outputs = await segmenter(img);

          let expected = [
            { score: 0.9916538596153259, label: "cat", mask: 58998 },
            { score: 0.9987397789955139, label: "remote", mask: 4164 },
            { score: 0.9994599223136902, label: "remote", mask: 2275 },
            { score: 0.9730215072631836, label: "couch", mask: 176980 },
            { score: 0.9993911385536194, label: "cat", mask: 52670 },
          ];

          let outputLabels = outputs.map((x) => x.label);
          let expectedLabels = expected.map((x) => x.label);

          expect(outputLabels).toHaveLength(expectedLabels.length);
          expect(outputLabels.sort()).toEqual(expectedLabels.sort());
        }

        await segmenter.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      models[1],
      async () => {
        let segmenter = await pipeline("image-segmentation", models[1]);
        let img = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/young-man-standing-and-leaning-on-car.jpg";

        // single
        {
          let outputs = await segmenter(img);

          let expected = [{ label: "Background" }, { label: "Hair" }, { label: "Upper-clothes" }, { label: "Pants" }, { label: "Left-shoe" }, { label: "Right-shoe" }, { label: "Face" }, { label: "Left-leg" }, { label: "Right-leg" }, { label: "Left-arm" }, { label: "Right-arm" }];

          let outputLabels = outputs.map((x) => x.label);
          let expectedLabels = expected.map((x) => x.label);

          expect(outputLabels).toHaveLength(expectedLabels.length);
          expect(outputLabels.sort()).toEqual(expectedLabels.sort());

          // check that all scores are null, and masks have correct dimensions
          for (let output of outputs) {
            expect(output.score).toBeNull();
            expect(output.mask.width).toEqual(970);
            expect(output.mask.height).toEqual(1455);
            expect(output.mask.channels).toEqual(1);
          }
        }

        await segmenter.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });

  describe("Zero-shot image classification", () => {
    // List all models which will be tested
    const models = ["Xenova/clip-vit-base-patch32"];

    it(
      models[0],
      async () => {
        let classifier = await pipeline("zero-shot-image-classification", models[0]);

        let url = "https://huggingface.co/datasets/mishig/sample_images/resolve/main/football-match.jpg";
        let urls = ["https://huggingface.co/datasets/mishig/sample_images/resolve/main/football-match.jpg", "https://huggingface.co/datasets/mishig/sample_images/resolve/main/airport.jpg", "https://huggingface.co/datasets/mishig/sample_images/resolve/main/savanna.jpg"];

        let classes = ["football", "airport", "animals"];

        // single
        {
          let output = await classifier(url, classes);

          let expected = [
            { score: 0.9719080924987793, label: "football" },
            { score: 0.022564826533198357, label: "animals" },
            { score: 0.005527070723474026, label: "airport" },
          ];
          compare(output, expected, 0.1);
        }

        // batched
        {
          let output = await classifier(urls, classes);

          let expected = [
            [
              { score: 0.9712504148483276, label: "football" },
              { score: 0.022469401359558105, label: "animals" },
              { score: 0.006280169822275639, label: "airport" },
            ],
            [
              { score: 0.997433602809906, label: "airport" },
              { score: 0.0016500800848007202, label: "animals" },
              { score: 0.0009163151844404638, label: "football" },
            ],
            [
              { score: 0.9851226806640625, label: "animals" },
              { score: 0.007516484707593918, label: "football" },
              { score: 0.007360846735537052, label: "airport" },
            ],
          ];
          compare(output, expected, 0.1);
        }
        await classifier.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });

  describe("Object detection", () => {
    // List all models which will be tested
    const models = ["Xenova/detr-resnet-50"];

    it(
      models[0],
      async () => {
        let detector = await pipeline("object-detection", models[0]);

        // TODO add batched test cases when supported
        let url = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg";
        let urls = ["https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/savanna.jpg"];

        // single + threshold
        {
          let output = await detector(url, {
            threshold: 0.9,
          });

          // let expected = [
          //     {
          //         "score": 0.9977124929428101,
          //         "label": "remote",
          //         "box": { "xmin": 41, "ymin": 70, "xmax": 176, "ymax": 118 }
          //     },
          //     {
          //         "score": 0.9984639883041382,
          //         "label": "remote",
          //         "box": { "xmin": 332, "ymin": 73, "xmax": 369, "ymax": 188 }
          //     },
          //     {
          //         "score": 0.9964856505393982,
          //         "label": "couch",
          //         "box": { "xmin": 0, "ymin": 1, "xmax": 639, "ymax": 474 }
          //     },
          //     {
          //         "score": 0.9988334774971008,
          //         "label": "cat",
          //         "box": { "xmin": 11, "ymin": 51, "xmax": 314, "ymax": 472 }
          //     },
          //     {
          //         "score": 0.9982513785362244,
          //         "label": "cat",
          //         "box": { "xmin": 345, "ymin": 22, "xmax": 640, "ymax": 371 }
          //     }
          // ]

          expect(output.length).toBeGreaterThan(0);
          for (let cls of output) {
            expect(typeof cls.score).toBe("number");
            expect(typeof cls.label).toBe("string");
            for (let key of ["xmin", "ymin", "xmax", "ymax"]) {
              expect(typeof cls.box[key]).toBe("number");
            }
          }
        }

        // batched + threshold + percentage
        {
          let output = await detector(urls, {
            threshold: 0.9,
            percentage: true,
          });
          // let expected = [[
          //     {
          //         score: 0.9991137385368347,
          //         label: 'zebra',
          //         box: { xmin: 0.65165576338768, ymin: 0.685152679681778, xmax: 0.723189502954483, ymax: 0.8801506459712982 }
          //     },
          //     {
          //         score: 0.998811662197113,
          //         label: 'zebra',
          //         box: { xmin: 0.20797613263130188, ymin: 0.6543092578649521, xmax: 0.4147692620754242, ymax: 0.9040975719690323 }
          //     },
          //     {
          //         score: 0.9707837104797363,
          //         label: 'giraffe',
          //         box: { xmin: 0.02498096227645874, ymin: 0.40549489855766296, xmax: 0.38669759035110474, ymax: 0.7895723879337311 }
          //     },
          //     {
          //         score: 0.9984336495399475,
          //         label: 'zebra',
          //         box: { xmin: 0.3540637195110321, ymin: 0.6370827257633209, xmax: 0.5765090882778168, ymax: 0.8480959832668304 }
          //     },
          //     {
          //         score: 0.9986463785171509,
          //         label: 'giraffe',
          //         box: { xmin: 0.6763969212770462, ymin: 0.25748637318611145, xmax: 0.974339172244072, ymax: 0.8684568107128143 }
          //     }
          // ]]

          expect(output).toHaveLength(urls.length); // Same number of inputs as outputs

          for (let i = 0; i < output.length; ++i) {
            expect(output[i].length).toBeGreaterThan(0);
            for (let cls of output[i]) {
              expect(typeof cls.score).toBe("number");
              expect(typeof cls.label).toBe("string");
              for (let key of ["xmin", "ymin", "xmax", "ymax"]) {
                expect(typeof cls.box[key]).toBe("number");
              }
            }
          }
        }

        await detector.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });

  describe("Zero-shot object detection", () => {
    // List all models which will be tested
    const models = ["Xenova/owlvit-base-patch32"];

    it(
      models[0],
      async () => {
        let detector = await pipeline("zero-shot-object-detection", models[0]);

        // single (default)
        {
          let url = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/astronaut.png";
          let candidate_labels = ["human face", "rocket", "helmet", "american flag"];

          let output = await detector(url, candidate_labels);

          // let expected = [
          //     {
          //         score: 0.24392342567443848,
          //         label: 'human face',
          //         box: { xmin: 180, ymin: 67, xmax: 274, ymax: 175 }
          //     },
          //     {
          //         score: 0.15129457414150238,
          //         label: 'american flag',
          //         box: { xmin: 0, ymin: 4, xmax: 106, ymax: 513 }
          //     },
          //     {
          //         score: 0.13649864494800568,
          //         label: 'helmet',
          //         box: { xmin: 277, ymin: 337, xmax: 511, ymax: 511 }
          //     },
          //     {
          //         score: 0.10262022167444229,
          //         label: 'rocket',
          //         box: { xmin: 352, ymin: -1, xmax: 463, ymax: 287 }
          //     }
          // ]

          expect(output.length).toBeGreaterThan(0);
          for (let cls of output) {
            expect(typeof cls.score).toBe("number");
            expect(typeof cls.label).toBe("string");
            for (let key of ["xmin", "ymin", "xmax", "ymax"]) {
              expect(typeof cls.box[key]).toBe("number");
            }
          }
        }

        // topk + threshold + percentage
        {
          let url = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/beach.png";
          let candidate_labels = ["hat", "book", "sunglasses", "camera"];

          let output = await detector(url, candidate_labels, {
            topk: 4,
            threshold: 0.05,
            percentage: true,
          });

          // let expected = [
          //     {
          //         score: 0.1606510728597641,
          //         label: 'sunglasses',
          //         box: { xmin: 347, ymin: 229, xmax: 429, ymax: 264 }
          //     },
          //     {
          //         score: 0.08935828506946564,
          //         label: 'hat',
          //         box: { xmin: 38, ymin: 174, xmax: 258, ymax: 364 }
          //     },
          //     {
          //         score: 0.08530698716640472,
          //         label: 'camera',
          //         box: { xmin: 187, ymin: 350, xmax: 260, ymax: 411 }
          //     },
          //     {
          //         score: 0.08349756896495819,
          //         label: 'book',
          //         box: { xmin: 261, ymin: 280, xmax: 494, ymax: 425 }
          //     }
          // ]

          expect(output.length).toBeGreaterThan(0);
          for (let cls of output) {
            expect(typeof cls.score).toBe("number");
            expect(typeof cls.label).toBe("string");
            for (let key of ["xmin", "ymin", "xmax", "ymax"]) {
              expect(typeof cls.box[key]).toBe("number");
            }
          }
        }

        await detector.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });

  describe("Image-to-image", () => {
    // List all models which will be tested
    const models = ["Xenova/swin2SR-classical-sr-x2-64"];

    it(
      models[0],
      async () => {
        let upscaler = await pipeline("image-to-image", models[0]);

        // Input is 3x3 => padded to 8x8 => upscaled to 16x16
        let url = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/pattern_3x3.png";

        // single
        {
          let outputs = await upscaler(url);
          expect(outputs.width).toEqual(16);
          expect(outputs.height).toEqual(16);
          expect(outputs.channels).toEqual(3);
          expect(outputs.data).toHaveLength(768);
        }

        // batched
        {
          let outputs = await upscaler([url, url]);
          expect(outputs).toHaveLength(2);
          for (let output of outputs) {
            expect(output.width).toEqual(16);
            expect(output.height).toEqual(16);
            expect(output.channels).toEqual(3);
            expect(output.data).toHaveLength(768);
          }
        }

        await upscaler.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });

  describe("Depth estimation", () => {
    // List all models which will be tested
    const models = ["Xenova/dpt-hybrid-midas"];

    it(
      models[0],
      async () => {
        let depth_estimator = await pipeline("depth-estimation", models[0]);

        let url = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg";

        // single
        {
          let { predicted_depth, depth } = await depth_estimator(url);
          compare(predicted_depth.dims, [384, 384]);
          expect(depth.width).toEqual(640);
          expect(depth.height).toEqual(480);
          expect(depth.channels).toEqual(1);
          expect(depth.data).toHaveLength(307200);
        }

        // batched
        {
          let outputs = await depth_estimator([url, url]);
          expect(outputs).toHaveLength(2);
          for (let output of outputs) {
            let { predicted_depth, depth } = output;
            compare(predicted_depth.dims, [384, 384]);
            expect(depth.width).toEqual(640);
            expect(depth.height).toEqual(480);
            expect(depth.channels).toEqual(1);
            expect(depth.data).toHaveLength(307200);
          }
        }

        await depth_estimator.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });

  describe("Document question answering", () => {
    // List all models which will be tested
    const models = ["Xenova/donut-base-finetuned-docvqa"];
    const image = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/invoice.png";
    const question = "What is the invoice number?";

    it(
      models[0],
      async () => {
        let qa_pipeline = await pipeline("document-question-answering", models[0]);

        // basic
        {
          let output = await qa_pipeline(image, question);
          let expected = [{ answer: "us-001" }];
          compare(output, expected);
        }

        await qa_pipeline.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
});
