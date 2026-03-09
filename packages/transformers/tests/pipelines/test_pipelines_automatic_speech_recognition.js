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

    describe("nemo-conformer-tdt", () => {
      const makeUnitPipe = () => {
        const calls = [];
        const model = {
          config: { model_type: "nemo-conformer-tdt" },
          async transcribe(_inputs, options) {
            calls.push(options);
            const result = { text: "hello world" };
            if (options.returnTimestamps) {
              result.utteranceTimestamp = [0, 0.08];
              result.words = [
                { text: "hello", startTime: 0, endTime: 0.04 },
                { text: "world", startTime: 0.04, endTime: 0.08 },
              ];
            }
            return result;
          },
          async dispose() {},
        };
        const processor = Object.assign(async () => ({ input_features: {} }), {
          feature_extractor: { config: { sampling_rate: 16000 } },
        });

        return {
          pipe: new AutomaticSpeechRecognitionPipeline({
            task: PIPELINE_ID,
            model,
            tokenizer: {},
            processor,
          }),
          calls,
        };
      };

      it("returns text when timestamps are disabled", async () => {
        const { pipe, calls } = makeUnitPipe();
        await expect(pipe(new Float32Array(16000), { return_timestamps: false })).resolves.toEqual({
          text: "hello world",
        });
        expect(calls).toHaveLength(1);
        expect(calls[0]).toMatchObject({
          returnTimestamps: false,
          returnWords: false,
          returnMetrics: false,
        });
      });

      it("returns sentence chunks when return_timestamps is true", async () => {
        const { pipe, calls } = makeUnitPipe();
        await expect(pipe(new Float32Array(16000), { return_timestamps: true })).resolves.toEqual({
          text: "hello world",
          chunks: [{ text: "hello world", timestamp: [0, 0.08] }],
        });
        expect(calls[0]).toMatchObject({
          returnTimestamps: true,
          returnWords: true,
          returnMetrics: false,
        });
      });

      it("returns word chunks when return_timestamps is 'word'", async () => {
        const { pipe, calls } = makeUnitPipe();
        await expect(pipe(new Float32Array(16000), { return_timestamps: "word" })).resolves.toEqual({
          text: "hello world",
          chunks: [
            { text: "hello", timestamp: [0, 0.04] },
            { text: "world", timestamp: [0.04, 0.08] },
          ],
        });
        expect(calls[0]).toMatchObject({
          returnTimestamps: true,
          returnWords: true,
          returnMetrics: false,
        });
      });
    });
  });
};
