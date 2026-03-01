import { pipeline, AutomaticSpeechRecognitionPipeline } from "../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";

const PIPELINE_ID = "automatic-speech-recognition";

export default () => {
  describe("Automatic Speech Recognition", () => {
    describe("whisper", () => {
      const model_id = "Xenova/tiny-random-WhisperForConditionalGeneration";
      const SAMPLING_RATE = 16000;
      const audios = [new Float32Array(SAMPLING_RATE).fill(0), Float32Array.from({ length: SAMPLING_RATE }, (_, i) => i / 16000)];
      const long_audios = [new Float32Array(SAMPLING_RATE * 60).fill(0), Float32Array.from({ length: SAMPLING_RATE * 60 }, (_, i) => (i % 1000) / 1000)];

      const max_new_tokens = 5;
      /** @type {AutomaticSpeechRecognitionPipeline} */
      let pipe;
      beforeAll(async () => {
        pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
      }, MAX_MODEL_LOAD_TIME);

      it("should be an instance of AutomaticSpeechRecognitionPipeline", () => {
        expect(pipe).toBeInstanceOf(AutomaticSpeechRecognitionPipeline);
      });

      describe("batch_size=1", () => {
        it(
          "default",
          async () => {
            const output = await pipe(audios[0], { max_new_tokens });
            const target = { text: "นะคะนะคะURURUR" };
            expect(output).toEqual(target);
          },
          MAX_TEST_EXECUTION_TIME,
        );
        it(
          "transcribe w/ return_timestamps=true",
          async () => {
            const output = await pipe(audios[0], { return_timestamps: true, max_new_tokens });
            const target = {
              text: " riceUR",
              chunks: [
                { timestamp: [0.72, 17.72], text: " rice" },
                { timestamp: [17.72, null], text: "UR" },
              ],
            };
            expect(output).toBeCloseToNested(target, 5);
          },
          MAX_TEST_EXECUTION_TIME,
        );
        // TODO add: transcribe w/ return_timestamps="word"
        // it(
        //   "transcribe w/ word-level timestamps",
        //   async () => {
        //     const output = await pipe(audios[0], { return_timestamps: "word", max_new_tokens });
        //     const target = [];
        //     expect(output).toBeCloseToNested(target, 5);
        //   },
        //   MAX_TEST_EXECUTION_TIME,
        // );
        it(
          "transcribe w/ language",
          async () => {
            const output = await pipe(audios[0], { language: "french", task: "transcribe", max_new_tokens });
            const target = { text: "นะคะนะคะURURUR" };
            expect(output).toEqual(target);
          },
          MAX_TEST_EXECUTION_TIME,
        );
        it(
          "translate",
          async () => {
            const output = await pipe(audios[0], { language: "french", task: "translate", max_new_tokens });
            const target = { text: "นะคะนะคะURURUR" };
            expect(output).toEqual(target);
          },
          MAX_TEST_EXECUTION_TIME,
        );
        it(
          "audio > 30 seconds",
          async () => {
            const output = await pipe(long_audios[0], { chunk_length_s: 30, stride_length_s: 5, max_new_tokens });
            const target = { text: "นะคะนะคะURURUR" };
            expect(output).toEqual(target);
          },
          MAX_TEST_EXECUTION_TIME,
        );
      });

      afterAll(async () => {
        await pipe?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });

    describe("wav2vec2", () => {
      const model_id = "Xenova/tiny-random-Wav2Vec2ForCTC-ONNX";
      const SAMPLING_RATE = 16000;
      const audios = [new Float32Array(SAMPLING_RATE).fill(0), Float32Array.from({ length: SAMPLING_RATE }, (_, i) => i / 16000)];
      const long_audios = [new Float32Array(SAMPLING_RATE * 60).fill(0), Float32Array.from({ length: SAMPLING_RATE * 60 }, (_, i) => (i % 1000) / 1000)];

      const max_new_tokens = 5;
      /** @type {AutomaticSpeechRecognitionPipeline} */
      let pipe;
      beforeAll(async () => {
        pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
      }, MAX_MODEL_LOAD_TIME);

      it("should be an instance of AutomaticSpeechRecognitionPipeline", () => {
        expect(pipe).toBeInstanceOf(AutomaticSpeechRecognitionPipeline);
      });

      describe("batch_size=1", () => {
        it(
          "default",
          async () => {
            const output = await pipe(audios[0], { max_new_tokens });
            const target = { text: "K" };
            expect(output).toEqual(target);
          },
          MAX_TEST_EXECUTION_TIME,
        );
      });

      afterAll(async () => {
        await pipe?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });

    describe("nemo-conformer-tdt (unit)", () => {
      const makeUnitPipe = (modelType = "nemo-conformer-tdt") => {
        const calls = [];
        const model = {
          config: { model_type: modelType },
          async transcribe(_inputs, options) {
            calls.push(options);
            const result = { text: "hello world" };
            if (options.return_timestamps) {
              result.utterance_timestamp = [0, 0.08];
              result.utterance_confidence = 0.95;
              result.confidence_scores = { token_avg: 0.95, word_avg: 0.94, overall_log_prob: -0.05 };
              if (options.return_words) {
                result.words = [
                  { text: "hello", start_time: 0, end_time: 0.04, confidence: 0.96 },
                  { text: "world", start_time: 0.04, end_time: 0.08, confidence: 0.93 },
                ];
              }
            }
            if (options.return_metrics) {
              result.metrics = { total_ms: 42, rtf: 0.01 };
            }
            return result;
          },
          async dispose() {},
        };

        const processor = Object.assign(async () => ({ input_features: {} }), {
          feature_extractor: { config: { sampling_rate: 16000 } },
        });
        const tokenizer = {};

        return {
          pipe: new AutomaticSpeechRecognitionPipeline({
            task: PIPELINE_ID,
            model,
            tokenizer,
            processor,
          }),
          calls,
        };
      };

      it("returns text and metrics when timestamps disabled", async () => {
        const { pipe, calls } = makeUnitPipe();
        const output = await pipe(new Float32Array(16000), { return_timestamps: false });
        expect(output).toEqual({ text: "hello world", metrics: { total_ms: 42, rtf: 0.01 } });
        expect(calls).toHaveLength(1);
        expect(calls[0]).toMatchObject({
          return_timestamps: false,
          return_words: false,
          return_metrics: true,
        });
      });

      it("returns full output with words when return_timestamps is true", async () => {
        const { pipe, calls } = makeUnitPipe();
        const output = await pipe(new Float32Array(16000), { return_timestamps: true });
        expect(output).toMatchObject({
          text: "hello world",
          utterance_timestamp: [0, 0.08],
          utterance_confidence: 0.95,
          words: [
            { text: "hello", start_time: 0, end_time: 0.04 },
            { text: "world", start_time: 0.04, end_time: 0.08 },
          ],
          confidence_scores: { token_avg: 0.95, word_avg: 0.94 },
          metrics: { total_ms: 42, rtf: 0.01 },
        });
        expect(calls[0]).toMatchObject({
          return_timestamps: true,
          return_words: true,
          return_metrics: true,
        });
      });

      it("treats return_timestamps 'word' as truthy (same as true)", async () => {
        const { pipe, calls } = makeUnitPipe();
        const output = await pipe(new Float32Array(16000), { return_timestamps: "word" });
        expect(output).toMatchObject({
          text: "hello world",
          utterance_timestamp: [0, 0.08],
          words: expect.any(Array),
          metrics: expect.any(Object),
        });
        expect(calls[0]).toMatchObject({
          return_timestamps: true,
          return_words: true,
          return_metrics: true,
        });
      });
    });
  });
};
