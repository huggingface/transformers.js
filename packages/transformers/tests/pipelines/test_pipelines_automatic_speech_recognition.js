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
      }, 2 * MAX_MODEL_LOAD_TIME);

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
                { text: " I", timestamp: [0, 1.36] },
                { text: " have", timestamp: [1.36, 1.68] },
                { text: " a", timestamp: [1.68, 1.94] },
                { text: " dream.", timestamp: [1.94, 2.54] },
                { text: " Good", timestamp: [3.74, 3.98] },
                { text: " one", timestamp: [3.98, 4.18] },
                { text: " day.", timestamp: [4.18, 4.8] },
                { text: " This", timestamp: [4.84, 6.56] },
                { text: " nation", timestamp: [6.56, 7.2] },
                { text: " will", timestamp: [7.2, 7.84] },
                { text: " rise", timestamp: [7.84, 8.3] },
                { text: " up.", timestamp: [8.3, 9.24] },
                { text: " Live", timestamp: [10.2, 10.56] },
                { text: " out", timestamp: [10.56, 10.98] },
                { text: " the", timestamp: [10.98, 11.02] },
                { text: " true", timestamp: [11.02, 11.3] },
                { text: " meaning", timestamp: [11.3, 11.58] },
                { text: " of", timestamp: [11.58, 11.86] },
                { text: " its", timestamp: [11.86, 12.06] },
                { text: " dream.", timestamp: [12.06, 12.58] },
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
      }, 2 * MAX_MODEL_LOAD_TIME);

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
                { text: " I", timestamp: [0, 1.4] },
                { text: " have", timestamp: [1.4, 1.7] },
                { text: " a", timestamp: [1.7, 1.88] },
                { text: " dream", timestamp: [1.88, 2.4] },
                { text: " that", timestamp: [2.4, 3.92] },
                { text: " one", timestamp: [3.92, 4.22] },
                { text: " day", timestamp: [4.22, 5.12] },
                { text: " this", timestamp: [5.12, 6.66] },
                { text: " nation", timestamp: [6.66, 7.34] },
                { text: " will", timestamp: [7.34, 7.86] },
                { text: " rise", timestamp: [7.86, 8.34] },
                { text: " up", timestamp: [8.34, 9.9] },
                { text: " and", timestamp: [9.9, 10.3] },
                { text: " live", timestamp: [10.3, 10.6] },
                { text: " out", timestamp: [10.6, 10.86] },
                { text: " the", timestamp: [10.86, 11.04] },
                { text: " true", timestamp: [11.04, 11.14] },
                { text: " meaning", timestamp: [11.38, 11.62] },
                { text: " of", timestamp: [11.62, 11.78] },
                { text: " its", timestamp: [11.78, 12.12] },
                { text: " creed.", timestamp: [12.12, 12.66] },
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
      }, 3 * MAX_MODEL_LOAD_TIME);

      it("should be an instance of AutomaticSpeechRecognitionPipeline", () => {
        expect(pipe).toBeInstanceOf(AutomaticSpeechRecognitionPipeline);
      });

      it(
        "transcribe w/ return_timestamps=true",
        async () => {
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
        },
        MAX_TEST_EXECUTION_TIME,
      );

      it(
        "transcribe w/ return_timestamps='word'",
        async () => {
          // Regression test: word-level timestamps must include content from all seek
          // passes, not just the first. Without the seek loop for return_token_timestamps,
          // the text was truncated after "the episode." (~23.6s), missing the final ~7s.
          const output = await pipe(audio, {
            return_timestamps: "word",
            chunk_length_s: 30,
            stride_length_s: 5,
            language: "en",
          });

          const target = {
            text: " everyday style. True classic delivers premium essentials built for real life. Grab yours at Target, Costco, or head to TrueClassic.com slash P4P. Get hooked up today. Now before we go, just want to give a big shout out to the CEO and founder Ryan brother for coming on our show and just showing some love. Now let's get back to the episode. I mean, like I said, we're going through that. We're losing stars. And then we kind of...",
            chunks: [
              { text: " everyday", timestamp: [0, 0.42] },
              { text: " style.", timestamp: [0.42, 0.84] },
              { text: " True", timestamp: [1.28, 1.56] },
              { text: " classic", timestamp: [1.56, 1.9] },
              { text: " delivers", timestamp: [1.9, 2.5] },
              { text: " premium", timestamp: [2.5, 3.02] },
              { text: " essentials", timestamp: [3.02, 3.54] },
              { text: " built", timestamp: [3.54, 3.84] },
              { text: " for", timestamp: [3.84, 4.08] },
              { text: " real", timestamp: [4.08, 4.4] },
              { text: " life.", timestamp: [4.4, 4.8] },
              { text: " Grab", timestamp: [5.64, 5.64] },
              { text: " yours", timestamp: [5.64, 6.04] },
              { text: " at", timestamp: [6.04, 6.36] },
              { text: " Target,", timestamp: [6.36, 6.86] },
              { text: " Costco,", timestamp: [7.18, 7.72] },
              { text: " or", timestamp: [8.26, 8.3] },
              { text: " head", timestamp: [8.3, 8.5] },
              { text: " to", timestamp: [8.5, 8.7] },
              { text: " TrueClassic", timestamp: [8.7, 9.32] },
              { text: ".com", timestamp: [9.32, 10.02] },
              { text: " slash", timestamp: [10.02, 10.78] },
              { text: " P4P.", timestamp: [10.78, 12.24] },
              { text: " Get", timestamp: [12.84, 13.06] },
              { text: " hooked", timestamp: [13.06, 13.3] },
              { text: " up", timestamp: [13.3, 13.52] },
              { text: " today.", timestamp: [13.52, 13.94] },
              { text: " Now", timestamp: [14.18, 14.24] },
              { text: " before", timestamp: [14.24, 14.46] },
              { text: " we", timestamp: [14.46, 14.62] },
              { text: " go,", timestamp: [14.62, 14.84] },
              { text: " just", timestamp: [15.04, 15.24] },
              { text: " want", timestamp: [15.24, 15.42] },
              { text: " to", timestamp: [15.42, 15.48] },
              { text: " give", timestamp: [15.48, 15.56] },
              { text: " a", timestamp: [15.56, 15.68] },
              { text: " big", timestamp: [15.68, 15.86] },
              { text: " shout", timestamp: [15.86, 16.1] },
              { text: " out", timestamp: [16.1, 16.28] },
              { text: " to", timestamp: [16.28, 16.74] },
              { text: " the", timestamp: [16.74, 16.9] },
              { text: " CEO", timestamp: [16.9, 17.52] },
              { text: " and", timestamp: [17.52, 17.86] },
              { text: " founder", timestamp: [17.86, 18.3] },
              { text: " Ryan", timestamp: [18.3, 18.56] },
              { text: " brother", timestamp: [18.72, 18.86] },
              { text: " for", timestamp: [18.86, 19.06] },
              { text: " coming", timestamp: [19.06, 19.2] },
              { text: " on", timestamp: [19.2, 19.36] },
              { text: " our", timestamp: [19.36, 19.5] },
              { text: " show", timestamp: [19.5, 19.74] },
              { text: " and", timestamp: [19.74, 20.4] },
              { text: " just", timestamp: [20.4, 20.6] },
              { text: " showing", timestamp: [20.6, 20.86] },
              { text: " some", timestamp: [20.86, 21.08] },
              { text: " love.", timestamp: [21.08, 21.28] },
              { text: " Now", timestamp: [21.46, 21.6] },
              { text: " let's", timestamp: [21.6, 22.36] },
              { text: " get", timestamp: [22.36, 22.46] },
              { text: " back", timestamp: [22.46, 22.68] },
              { text: " to", timestamp: [22.68, 23.06] },
              { text: " the", timestamp: [23.06, 23.2] },
              { text: " episode.", timestamp: [23.2, 23.58] },
              { text: " I", timestamp: [23.6, 24.42] },
              { text: " mean,", timestamp: [24.42, 24.62] },
              { text: " like", timestamp: [25.08, 25.42] },
              { text: " I", timestamp: [25.42, 25.54] },
              { text: " said,", timestamp: [25.54, 25.7] },
              { text: " we're", timestamp: [25.9, 26.12] },
              { text: " going", timestamp: [26.12, 26.34] },
              { text: " through", timestamp: [26.34, 26.58] },
              { text: " that.", timestamp: [26.58, 26.86] },
              { text: " We're", timestamp: [27.04, 27.34] },
              { text: " losing", timestamp: [27.34, 27.54] },
              { text: " stars.", timestamp: [27.54, 28.06] },
              { text: " And", timestamp: [28.96, 29.3] },
              { text: " then", timestamp: [29.3, 29.38] },
              { text: " we", timestamp: [29.38, 29.58] },
              { text: " kind", timestamp: [29.58, 29.84] },
              { text: " of", timestamp: [29.84, 29.98] },
              { text: "...", timestamp: [29.98, 29.98] },
            ],
          };
          expect(output).toBeCloseToNested(target, 1);
        },
        // Word-level timestamp inference can exceed 60 seconds on shared CI runners.
        2 * MAX_TEST_EXECUTION_TIME,
      );
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

    describe("moonshine", () => {
      it("decodes generated tokens with the pipeline tokenizer", async () => {
        const sampling_rate = 16000;
        const generated_tokens = [[1n, 2n, 3n]];
        let decode_args;

        const processor = Object.assign(async (audio) => ({ input_values: audio }), {
          feature_extractor: { config: { sampling_rate } },
        });
        const tokenizer = {
          batch_decode(outputs, options) {
            decode_args = { outputs, options };
            return ["decoded transcript"];
          },
        };
        const model = {
          config: { model_type: "moonshine" },
          async generate() {
            return generated_tokens;
          },
        };
        const pipe = new AutomaticSpeechRecognitionPipeline({
          task: PIPELINE_ID,
          model,
          tokenizer,
          processor,
        });

        const output = await pipe(new Float32Array(sampling_rate));

        expect(output).toEqual({ text: "decoded transcript" });
        expect(decode_args).toEqual({
          outputs: generated_tokens,
          options: { skip_special_tokens: true },
        });
      });
    });
  });
};
