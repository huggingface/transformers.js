import { pipeline, AutomaticSpeechRecognitionPipeline } from "../../src/transformers.js";
import { load_cached_audio } from "../asset_cache.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";

const PIPELINE_ID = "automatic-speech-recognition";

export default () => {
  describe("Automatic Speech Recognition", () => {
    describe("whisper (tiny-random)", () => {
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
              text: "�를 ис",
              chunks: [{ timestamp: [28.96, null], text: "�를 ис" }],
            };
            expect(output).toBeCloseToNested(target, 5);
          },
          MAX_TEST_EXECUTION_TIME,
        );
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

    describe("whisper", () => {
      const model_id = "onnx-community/whisper-tiny_timestamped";

      /** @type {AutomaticSpeechRecognitionPipeline} */
      let pipe;
      /** @type {Float64Array} */
      let audio;
      beforeAll(async () => {
        pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
        audio = await load_cached_audio("mlk");
      }, MAX_MODEL_LOAD_TIME);

      it("should be an instance of AutomaticSpeechRecognitionPipeline", () => {
        expect(pipe).toBeInstanceOf(AutomaticSpeechRecognitionPipeline);
      });

      describe("batch_size=1", () => {
        it(
          "default",
          async () => {
            const output = await pipe(audio, { language: "en", task: "transcribe" });
            const target = {
              text: " I have a dream. Good one day. This nation will rise up. Live out the true meaning of its dream.",
            };
            expect(output).toEqual(target);
          },
          MAX_TEST_EXECUTION_TIME,
        );
        it(
          "transcribe w/ return_timestamps=true",
          async () => {
            const output = await pipe(audio, { return_timestamps: true, language: "en", task: "transcribe" });
            const target = {
              text: " I have a dream. Good one day. This nation will rise up. Live out the true meaning of its dream.",
              chunks: [
                { timestamp: [0, 2.74], text: " I have a dream." },
                { timestamp: [2.74, 5.34], text: " Good one day." },
                { timestamp: [5.34, 9.24], text: " This nation will rise up." },
                { timestamp: [9.24, 12.58], text: " Live out the true meaning of its dream." },
              ],
            };
            expect(output).toBeCloseToNested(target, 1);
          },
          MAX_TEST_EXECUTION_TIME,
        );
        it(
          "transcribe w/ return_timestamps='word'",
          async () => {
            const output = await pipe(audio, { return_timestamps: "word", language: "en", task: "transcribe" });
            const target = {
              text: " I have a dream. Good one day. This nation will rise up. Live out the true meaning of its dream.",
              chunks: [
                { text: " I", timestamp: [1.36, 1.68] },
                { text: " have", timestamp: [1.68, 1.94] },
                { text: " a", timestamp: [1.94, 2.52] },
                { text: " dream.", timestamp: [2.52, 2.74] },
                { text: " Good", timestamp: [3.98, 4.18] },
                { text: " one", timestamp: [4.18, 4.78] },
                { text: " day.", timestamp: [4.78, 4.84] },
                { text: " This", timestamp: [6.56, 7.2] },
                { text: " nation", timestamp: [7.2, 7.84] },
                { text: " will", timestamp: [7.84, 8.3] },
                { text: " rise", timestamp: [8.3, 9.24] },
                { text: " up.", timestamp: [9.84, 9.86] },
                { text: " Live", timestamp: [10.56, 10.98] },
                { text: " out", timestamp: [10.98, 11.02] },
                { text: " the", timestamp: [11.02, 11.3] },
                { text: " true", timestamp: [11.3, 11.58] },
                { text: " meaning", timestamp: [11.58, 11.86] },
                { text: " of", timestamp: [11.86, 12.06] },
                { text: " its", timestamp: [12.06, 12.56] },
                { text: " dream.", timestamp: [12.56, 12.58] },
              ],
            };
            expect(output).toBeCloseToNested(target, 1);
          },
          MAX_TEST_EXECUTION_TIME,
        );
        it(
          "translate",
          async () => {
            const output = await pipe(audio, { language: "french", task: "translate" });
            const target = {
              text: " I have a dream. Good one day. This nation will rise up. Live out the true meaning of its dream.",
            };
            expect(output).toEqual(target);
          },
          MAX_TEST_EXECUTION_TIME,
        );
      });

      afterAll(async () => {
        await pipe?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });

    describe("whisper (en-only)", () => {
      const model_id = "onnx-community/whisper-tiny.en_timestamped";

      /** @type {AutomaticSpeechRecognitionPipeline} */
      let pipe;
      /** @type {Float64Array} */
      let audio;
      beforeAll(async () => {
        pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
        audio = await load_cached_audio("mlk");
      }, MAX_MODEL_LOAD_TIME);

      it("should be an instance of AutomaticSpeechRecognitionPipeline", () => {
        expect(pipe).toBeInstanceOf(AutomaticSpeechRecognitionPipeline);
      });

      describe("batch_size=1", () => {
        it(
          "default",
          async () => {
            const output = await pipe(audio);
            const target = {
              text: " I have a dream that one day this nation will rise up live out the true meaning of its creed",
            };
            expect(output).toEqual(target);
          },
          MAX_TEST_EXECUTION_TIME,
        );
        it(
          "transcribe w/ return_timestamps=true",
          async () => {
            const output = await pipe(audio, { return_timestamps: true });
            const target = {
              text: " I have a dream that one day this nation will rise up and live out the true meaning of its creed.",
              chunks: [
                {
                  timestamp: [0, 11.14],
                  text: " I have a dream that one day this nation will rise up and live out the true",
                },
                { timestamp: [11.14, 14.18], text: " meaning of its creed." },
              ],
            };
            expect(output).toBeCloseToNested(target, 1);
          },
          MAX_TEST_EXECUTION_TIME,
        );
        it(
          "transcribe w/ return_timestamps='word'",
          async () => {
            const output = await pipe(audio, { return_timestamps: "word" });
            const target = {
              text: " I have a dream that one day this nation will rise up and live out the true meaning of its creed.",
              chunks: [
                { text: " I", timestamp: [1.4, 1.7] },
                { text: " have", timestamp: [1.7, 1.88] },
                { text: " a", timestamp: [1.88, 2.4] },
                { text: " dream", timestamp: [2.4, 3.92] },
                { text: " that", timestamp: [3.92, 4.22] },
                { text: " one", timestamp: [4.22, 5.12] },
                { text: " day", timestamp: [5.12, 6.66] },
                { text: " this", timestamp: [6.66, 7.34] },
                { text: " nation", timestamp: [7.34, 7.86] },
                { text: " will", timestamp: [7.86, 8.34] },
                { text: " rise", timestamp: [8.34, 9.9] },
                { text: " up", timestamp: [9.9, 10.3] },
                { text: " and", timestamp: [10.3, 10.6] },
                { text: " live", timestamp: [10.6, 10.86] },
                { text: " out", timestamp: [10.86, 11.04] },
                { text: " the", timestamp: [11.04, 11.14] },
                { text: " true", timestamp: [11.34, 11.38] },
                { text: " meaning", timestamp: [11.62, 11.78] },
                { text: " of", timestamp: [11.78, 12.12] },
                { text: " its", timestamp: [12.12, 12.64] },
                { text: " creed.", timestamp: [12.64, 13.6] },
              ],
            };
            expect(output).toBeCloseToNested(target, 1);
          },
          MAX_TEST_EXECUTION_TIME,
        );
      });

      afterAll(async () => {
        await pipe?.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });

    describe("whisper (base)", () => {
      const model_id = "onnx-community/whisper-base_timestamped";

      /** @type {AutomaticSpeechRecognitionPipeline} */
      let pipe;
      /** @type {Float64Array} */
      let audio;
      beforeAll(async () => {
        pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
        audio = await load_cached_audio("whisper_1");
      }, MAX_MODEL_LOAD_TIME);

      it("should be an instance of AutomaticSpeechRecognitionPipeline", () => {
        expect(pipe).toBeInstanceOf(AutomaticSpeechRecognitionPipeline);
      });

      it("transcribe w/ return_timestamps='word'", async () => {
        const output = await pipe(audio, {
          return_timestamps: true,
          chunk_length_s: 30,
          stride_length_s: 5,
          language: "en",
        });
        // Python output for reference:
        // const target = {
        //   text: " everyday style. True classic delivers premium essentials built for real life. Grab yours at Target, Costco, or head to TrueClassic.com slash P4P. Get hooked up today. Now before we go, just want to give a big shout out to the CEO and founder Ryan brother for coming on our show and just showing some love. Now let's get back to the episode. I mean, like I said, we're going through that. We're losing stars. And then we kind of...",
        //   chunks: [
        //     { timestamp: [0.0, 4.8], text: " everyday style. True classic delivers premium essentials built for real life." },
        //     { timestamp: [5.36, 14.0], text: " Grab yours at Target, Costco, or head to TrueClassic.com slash P4P. Get hooked up today." },
        //     { timestamp: [14.0, 18.56], text: " Now before we go, just want to give a big shout out to the CEO and founder Ryan" },
        //     { timestamp: [18.56, 23.6], text: " brother for coming on our show and just showing some love. Now let's get back to the episode." },
        //     { timestamp: [24.24, 23.6], text: "" },
        //     { timestamp: [27.02, 29.18], text: " I mean, like I said, we're going through that. We're losing stars." },
        //     { timestamp: [29.18, 30.54], text: " And then we kind of..." },
        //   ],
        // };

        const target = {
          text: " everyday style. True classic delivers premium essentials built for real life. Grab yours at Target, Costco, or head to TrueClassic.com slash P4P. Get hooked up today. Now before we go, just want to give a big shout out to the CEO and founder Ryan brother for coming on our show and just showing some love. Now let's get back to the episode. I mean, like I said, we're going through that. We're losing stars. And then we kind of...",
          chunks: [
            { timestamp: [0, 4.8], text: " everyday style. True classic delivers premium essentials built for real life." },
            { timestamp: [5.36, 14], text: " Grab yours at Target, Costco, or head to TrueClassic.com slash P4P. Get hooked up today." },
            { timestamp: [14, 18.56], text: " Now before we go, just want to give a big shout out to the CEO and founder Ryan" },
            { timestamp: [18.56, 23.6], text: " brother for coming on our show and just showing some love. Now let's get back to the episode." },
            { timestamp: [23.6, 27.02], text: " I mean, like I said, we're going through that." },
            { timestamp: [27.02, 29.18], text: " We're losing stars." },
            { timestamp: [29.18, 30], text: " And then we kind of..." },
          ],
        };

        expect(output).toBeCloseToNested(target, 1);
      });
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
  });
};
