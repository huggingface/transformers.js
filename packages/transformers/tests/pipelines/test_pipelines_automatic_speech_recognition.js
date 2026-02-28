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
            return {
              text: "hello world",
              token_ids: [1, 2],
              token_timestamps: [
                [0, 0.04],
                [0.04, 0.08],
              ],
              word_timestamps: [
                { text: "hello", timestamp: [0, 0.04] },
                { text: "world", timestamp: [0.04, 0.08] },
              ],
              utterance_timestamp: [0, 0.08],
            };
          },
          async dispose() {},
        };

        const processor = Object.assign(async () => ({ input_features: {} }), {
          feature_extractor: { config: { sampling_rate: 16000 } },
        });
        const tokenizer = {
          decode(ids) {
            const idArray = Array.isArray(ids) ? ids : [ids];
            return idArray
              .map((id) => {
                if (id === 1 || id === 1n) return " hello";
                if (id === 2 || id === 2n) return " world";
                return "";
              })
              .join("");
          },
        };

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

      it("dispatches to nemo-conformer-tdt path", async () => {
        const { pipe, calls } = makeUnitPipe();
        const output = await pipe(new Float32Array(16000), { return_timestamps: false });
        expect(output).toEqual({ text: "hello world" });
        expect(calls).toHaveLength(1);
      });

      it("default timestamps use word granularity", async () => {
        const { pipe, calls } = makeUnitPipe();
        const output = await pipe(new Float32Array(16000), { return_timestamps: true });
        expect(output).toEqual({
          text: "hello world",
          chunks: [
            { text: "hello", timestamp: [0, 0.04] },
            { text: "world", timestamp: [0.04, 0.08] },
          ],
        });
        expect(calls[0]).toMatchObject({
          return_word_timestamps: true,
          return_token_timestamps: false,
          return_utterance_timestamp: false,
        });
      });

      it("supports utterance granularity", async () => {
        const { pipe } = makeUnitPipe();
        const output = await pipe(new Float32Array(16000), {
          return_timestamps: true,
          timestamp_granularity: "utterance",
        });
        expect(output).toEqual({
          text: "hello world",
          chunks: [{ text: "hello world", timestamp: [0, 0.08] }],
        });
      });

      it("supports token granularity", async () => {
        const { pipe } = makeUnitPipe();
        const output = await pipe(new Float32Array(16000), {
          return_timestamps: true,
          timestamp_granularity: "token",
        });
        expect(output).toEqual({
          text: "hello world",
          chunks: [
            { text: " hello", timestamp: [0, 0.04] },
            { text: " world", timestamp: [0.04, 0.08] },
          ],
        });
      });

      it("supports all granularities at once", async () => {
        const { pipe } = makeUnitPipe();
        const output = await pipe(new Float32Array(16000), {
          return_timestamps: true,
          timestamp_granularity: "all",
        });
        expect(output).toEqual({
          text: "hello world",
          chunks: [
            { text: "hello", timestamp: [0, 0.04] },
            { text: "world", timestamp: [0.04, 0.08] },
          ],
          tokens: [
            { text: " hello", timestamp: [0, 0.04] },
            { text: " world", timestamp: [0.04, 0.08] },
          ],
          utterance: [0, 0.08],
        });
      });

      it("throws for invalid timestamp granularity", async () => {
        const { pipe } = makeUnitPipe();
        await expect(
          pipe(new Float32Array(16000), {
            return_timestamps: true,
            timestamp_granularity: "frame",
          }),
        ).rejects.toThrow("Invalid `timestamp_granularity`");
      });
    });
  });
};
