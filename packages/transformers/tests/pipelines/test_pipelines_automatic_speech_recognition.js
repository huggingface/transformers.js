import { pipeline, AutomaticSpeechRecognitionPipeline, Tensor } from "../../src/transformers.js";
import { NEMO_FEATURE_OUTPUT_OWNERSHIP } from "../../src/models/nemo_conformer_tdt/feature_extraction_nemo_conformer_tdt.js";

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
      const withNemoTensorOwnership = (value, cacheOwnsTensors) => {
        Object.defineProperty(value, NEMO_FEATURE_OUTPUT_OWNERSHIP, {
          value: cacheOwnsTensors,
          enumerable: false,
          configurable: true,
        });
        return value;
      };

      const makeUnitPipe = (modelType = "nemo-conformer-tdt") => {
        const calls = [];
        const model = {
          config: { model_type: modelType },
          async transcribe(_inputs, options) {
            calls.push(options);
            const result = { text: "hello world" };
            if (options.returnTimestamps) {
              result.utteranceTimestamp = [0, 0.08];
              result.confidence = { utterance: 0.95, wordAverage: 0.94, averageLogProb: -0.05 };
              if (options.returnWords) {
                result.words = [
                  { text: "hello", startTime: 0, endTime: 0.04, confidence: 0.96 },
                  { text: "world", startTime: 0.04, endTime: 0.08, confidence: 0.93 },
                ];
              }
            }
            if (options.returnMetrics) {
              result.metrics = { totalMs: 42, rtf: 0.01, rtfX: 100 };
            }
            return result;
          },
          async dispose() {},
        };

        const processor = Object.assign(async () => ({ input_features: {} }), {
          feature_extractor: { config: { sampling_rate: 16000 } },
        });
        const tokenizer = {
          decode(ids) {
            const pieces = {
              1: "hello",
              2: "world",
              3: "again",
              4: "today",
            };
            return ids.map((id) => pieces[id] ?? "").filter(Boolean).join(" ");
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

      it("returns text when timestamps disabled", async () => {
        const { pipe, calls } = makeUnitPipe();
        const output = await pipe(new Float32Array(16000), { return_timestamps: false });
        expect(output).toEqual({ text: "hello world" });
        expect(calls).toHaveLength(1);
        expect(calls[0]).toMatchObject({
          returnTimestamps: false,
          returnWords: false,
          returnMetrics: false,
        });
      });

      it("returns timestamped chunks when return_timestamps is true", async () => {
        const { pipe, calls } = makeUnitPipe();
        const output = await pipe(new Float32Array(16000), { return_timestamps: true });
        expect(output).toEqual({
          text: "hello world",
          chunks: [
            { text: "hello world", timestamp: [0, 0.08] },
          ],
        });
        expect(calls).toHaveLength(1);
        expect(calls[0]).toMatchObject({
          returnTimestamps: true,
          returnWords: true,
          returnMetrics: false,
        });
      });

      it("returns word chunks when return_timestamps is 'word'", async () => {
        const { pipe, calls } = makeUnitPipe();
        const output = await pipe(new Float32Array(16000), { return_timestamps: "word" });
        expect(output).toEqual({
          text: "hello world",
          chunks: [
            { text: "hello", timestamp: [0, 0.04] },
            { text: "world", timestamp: [0.04, 0.08] },
          ],
        });
        expect(calls).toHaveLength(1);
        expect(calls[0]).toMatchObject({
          returnTimestamps: true,
          returnWords: true,
          returnMetrics: false,
        });
      });

      it("builds conservative sentence chunks from Nemo word timestamps", async () => {
        const model = {
          config: { model_type: "nemo-conformer-tdt" },
          async transcribe() {
            return {
              text: "Hello. World again. U.S. Report update.",
              utteranceTimestamp: [0, 2.4],
              words: [
                { text: "Hello.", startTime: 0, endTime: 0.4 },
                { text: "World", startTime: 0.5, endTime: 0.8 },
                { text: "again.", startTime: 0.8, endTime: 1.1 },
                { text: "U.S.", startTime: 1.2, endTime: 1.5 },
                { text: "Report", startTime: 1.6, endTime: 2.0 },
                { text: "update.", startTime: 2.0, endTime: 2.4 },
              ],
            };
          },
          async dispose() {},
        };
        const processor = Object.assign(async () => ({ input_features: {} }), {
          feature_extractor: { config: { sampling_rate: 16000 } },
        });
        const pipe = new AutomaticSpeechRecognitionPipeline({
          task: PIPELINE_ID,
          model,
          tokenizer: {},
          processor,
        });

        const output = await pipe(new Float32Array(16000), { return_timestamps: true });
        expect(output).toEqual({
          text: "Hello. World again. U.S. Report update.",
          chunks: [
            { text: "Hello.", timestamp: [0, 0.4] },
            { text: "World again.", timestamp: [0.5, 1.1] },
            { text: "U.S. Report update.", timestamp: [1.2, 2.4] },
          ],
        });
      });

      it("uses explicit chunk_length_s as a bounded sentence window size override", async () => {
        const calls = [];
        const outputsByOffset = new Map([
          [0, {
            text: "Alpha. Beta. Carry",
            words: [
              { text: "Alpha.", startTime: 0, endTime: 1 },
              { text: "Beta.", startTime: 17, endTime: 18 },
              { text: "Carry", startTime: 19.95, endTime: 20 },
            ],
          }],
          [19.95, {
            text: "Carry on. Gamma",
            words: [
              { text: "Carry", startTime: 19.95, endTime: 20 },
              { text: "on.", startTime: 20, endTime: 20.5 },
              { text: "Gamma", startTime: 37.9, endTime: 38 },
            ],
          }],
          [37.9, {
            text: "Gamma. Tail resumes. Omega.",
            words: [
              { text: "Gamma.", startTime: 37.9, endTime: 39 },
              { text: "Tail", startTime: 39.2, endTime: 39.6 },
              { text: "resumes.", startTime: 39.6, endTime: 40.1 },
              { text: "Omega.", startTime: 40.1, endTime: 40.45 },
            ],
          }],
        ]);
        const model = {
          config: { model_type: "nemo-conformer-tdt" },
          async transcribe(_inputs, options) {
            calls.push(options);
            const item = outputsByOffset.get(options.timeOffset);
            if (!item) {
              throw new Error(`Unexpected timeOffset ${options.timeOffset}`);
            }
            return {
              text: item.text,
              utteranceTimestamp: [item.words[0].startTime, item.words[item.words.length - 1].endTime],
              words: item.words,
            };
          },
          async dispose() {},
        };
        const processor = Object.assign(async () => ({ input_features: {} }), {
          feature_extractor: { config: { sampling_rate: 16000 } },
        });
        const tokenizer = {
          decode(ids) {
            const pieces = {
              1: "hello",
              2: "world",
              3: "again",
              4: "today",
            };
            return ids.map((id) => pieces[id] ?? "").filter(Boolean).join(" ");
          },
        };
        const pipe = new AutomaticSpeechRecognitionPipeline({
          task: PIPELINE_ID,
          model,
          tokenizer,
          processor,
        });

        const output = await pipe(new Float32Array(40.5 * 16000), {
          return_timestamps: "word",
          chunk_length_s: 2,
        });

        expect(output).toEqual({
          text: "Alpha. Beta. Carry on. Gamma. Tail resumes. Omega.",
          chunks: [
            { text: "Alpha.", timestamp: [0, 1] },
            { text: "Beta.", timestamp: [17, 18] },
            { text: "Carry", timestamp: [19.95, 20] },
            { text: "on.", timestamp: [20, 20.5] },
            { text: "Gamma.", timestamp: [37.9, 39] },
            { text: "Tail", timestamp: [39.2, 39.6] },
            { text: "resumes.", timestamp: [39.6, 40.1] },
            { text: "Omega.", timestamp: [40.1, 40.45] },
          ],
        });
        expect(calls).toHaveLength(3);
        expect(calls.map((x) => x.timeOffset)).toEqual([0, 19.95, 37.9]);
        expect(calls[0]).toMatchObject({
          returnTimestamps: true,
          returnWords: true,
          returnMetrics: false,
          timeOffset: 0,
        });
        expect(calls[1]).toMatchObject({
          returnTimestamps: true,
          returnWords: true,
          returnMetrics: false,
          timeOffset: 19.95,
        });
        expect(calls[2]).toMatchObject({
          returnTimestamps: true,
          returnWords: true,
          returnMetrics: false,
          timeOffset: 37.9,
        });
      });

      it("replaces boundary-truncated sentences with the longer retranscribed sentence", async () => {
        const calls = [];
        const outputsByOffset = new Map([
          [0, {
            text: "Alpha. Beta. It won't run away, and it won't come to life.",
            words: [
              { text: "Alpha.", startTime: 0, endTime: 1 },
              { text: "Beta.", startTime: 11, endTime: 12 },
              { text: "It", startTime: 17.2, endTime: 17.5 },
              { text: "won't", startTime: 17.5, endTime: 17.9 },
              { text: "run", startTime: 17.9, endTime: 18.2 },
              { text: "away,", startTime: 18.2, endTime: 18.6 },
              { text: "and", startTime: 18.6, endTime: 18.8 },
              { text: "it", startTime: 18.8, endTime: 19.0 },
              { text: "won't", startTime: 19.0, endTime: 19.3 },
              { text: "come", startTime: 19.3, endTime: 19.5 },
              { text: "to", startTime: 19.5, endTime: 19.65 },
              { text: "life.", startTime: 19.65, endTime: 19.8 },
            ],
          }],
          [17.2, {
            text: "It won't run away, and it won't come to life until someone finds it. Omega.",
            words: [
              { text: "It", startTime: 17.2, endTime: 17.5 },
              { text: "won't", startTime: 17.5, endTime: 17.9 },
              { text: "run", startTime: 17.9, endTime: 18.2 },
              { text: "away,", startTime: 18.2, endTime: 18.6 },
              { text: "and", startTime: 18.6, endTime: 18.8 },
              { text: "it", startTime: 18.8, endTime: 19.0 },
              { text: "won't", startTime: 19.0, endTime: 19.3 },
              { text: "come", startTime: 19.3, endTime: 19.5 },
              { text: "to", startTime: 19.5, endTime: 19.65 },
              { text: "life", startTime: 19.65, endTime: 19.95 },
              { text: "until", startTime: 19.95, endTime: 20.4 },
              { text: "someone", startTime: 20.4, endTime: 21.0 },
              { text: "finds", startTime: 21.0, endTime: 21.5 },
              { text: "it.", startTime: 21.5, endTime: 22.0 },
              { text: "Omega.", startTime: 28, endTime: 29 },
            ],
          }],
        ]);
        const model = {
          config: { model_type: "nemo-conformer-tdt" },
          async transcribe(_inputs, options) {
            calls.push(options);
            const item = outputsByOffset.get(options.timeOffset);
            if (!item) {
              throw new Error(`Unexpected timeOffset ${options.timeOffset}`);
            }
            return {
              text: item.text,
              utteranceTimestamp: [item.words[0].startTime, item.words[item.words.length - 1].endTime],
              words: item.words,
            };
          },
          async dispose() {},
        };
        const processor = Object.assign(async () => ({ input_features: {} }), {
          feature_extractor: { config: { sampling_rate: 16000 } },
        });
        const pipe = new AutomaticSpeechRecognitionPipeline({
          task: PIPELINE_ID,
          model,
          tokenizer: {},
          processor,
        });

        const output = await pipe(new Float32Array(Math.ceil(31 * 16000)), {
          return_timestamps: true,
          chunk_length_s: 20,
        });

        expect(output).toEqual({
          text: "Alpha. Beta. It won't run away, and it won't come to life until someone finds it. Omega.",
          chunks: [
            { text: "Alpha.", timestamp: [0, 1] },
            { text: "Beta.", timestamp: [11, 12] },
            { text: "It won't run away, and it won't come to life until someone finds it.", timestamp: [17.2, 22] },
            { text: "Omega.", timestamp: [28, 29] },
          ],
        });
        expect(calls.map((x) => x.timeOffset)).toEqual([0, 17.2]);
      });

      it("retranscribes the dropped last sentence from its start without stale carry", async () => {
        const calls = [];
        const outputsByOffset = new Map([
          [0, {
            text: "Alpha. The pressure gauge mark. He watched as the fruit",
            words: [
              { text: "Alpha.", startTime: 0, endTime: 1 },
              { text: "The", startTime: 16.8, endTime: 17.0 },
              { text: "pressure", startTime: 17.0, endTime: 17.4 },
              { text: "gauge", startTime: 17.4, endTime: 17.76 },
              { text: "mark.", startTime: 17.76, endTime: 18.56 },
              { text: "He", startTime: 18.56, endTime: 18.72 },
              { text: "watched", startTime: 18.72, endTime: 18.96 },
              { text: "as", startTime: 18.96, endTime: 19.04 },
              { text: "the", startTime: 19.04, endTime: 19.2 },
              { text: "fruit", startTime: 19.2, endTime: 19.36 },
            ],
          }],
          [18.56, {
            text: "He watched as the fluid.",
            words: [
              { text: "He", startTime: 18.56, endTime: 18.72 },
              { text: "watched", startTime: 18.72, endTime: 19.12 },
              { text: "as", startTime: 19.12, endTime: 19.28 },
              { text: "the", startTime: 19.28, endTime: 19.36 },
              { text: "fluid.", startTime: 19.36, endTime: 20 },
            ],
          }],
        ]);
        const model = {
          config: { model_type: "nemo-conformer-tdt" },
          async transcribe(_inputs, options) {
            calls.push(options);
            const item = outputsByOffset.get(options.timeOffset);
            if (!item) {
              throw new Error(`Unexpected timeOffset ${options.timeOffset}`);
            }
            return {
              text: item.text,
              utteranceTimestamp: [item.words[0].startTime, item.words[item.words.length - 1].endTime],
              words: item.words,
            };
          },
          async dispose() {},
        };
        const processor = Object.assign(async () => ({ input_features: {} }), {
          feature_extractor: { config: { sampling_rate: 16000 } },
        });
        const pipe = new AutomaticSpeechRecognitionPipeline({
          task: PIPELINE_ID,
          model,
          tokenizer: {},
          processor,
        });

        const output = await pipe(new Float32Array(Math.ceil(21 * 16000)), {
          return_timestamps: "word",
          chunk_length_s: 20,
        });

        expect(output).toEqual({
          text: "Alpha. The pressure gauge mark. He watched as the fluid.",
          chunks: [
            { text: "Alpha.", timestamp: [0, 1] },
            { text: "The", timestamp: [16.8, 17] },
            { text: "pressure", timestamp: [17, 17.4] },
            { text: "gauge", timestamp: [17.4, 17.76] },
            { text: "mark.", timestamp: [17.76, 18.56] },
            { text: "He", timestamp: [18.56, 18.72] },
            { text: "watched", timestamp: [18.72, 19.12] },
            { text: "as", timestamp: [19.12, 19.28] },
            { text: "the", timestamp: [19.28, 19.36] },
            { text: "fluid.", timestamp: [19.36, 20] },
          ],
        });
        expect(calls.map((x) => x.timeOffset)).toEqual([0, 18.56]);
      });

      it("reconstructs windowed Nemo text from merged words when token decode drops spaces", async () => {
        const calls = [];
        const model = {
          config: { model_type: "nemo-conformer-tdt" },
          async transcribe(_inputs, options) {
            calls.push(options);
            if (options.timeOffset === 0) {
              return {
                text: "score. 48-year-old",
                words: [
                  { text: "score.", startTime: 0, endTime: 0.4 },
                  { text: "48-year-old", startTime: 0.5, endTime: 1.3 },
                ],
              };
            }
            return {
              text: "with 0.5",
              words: [
                { text: "with", startTime: 1.4, endTime: 1.7 },
                { text: "0.5", startTime: 1.8, endTime: 2.05 },
              ],
            };
          },
          async dispose() {},
        };
        const processor = Object.assign(async () => ({ input_features: {} }), {
          feature_extractor: { config: { sampling_rate: 16000 } },
        });
        const tokenizer = {
          decode(ids) {
            const pieces = {
              1: "score",
              2: ".",
              3: "48",
              4: "-",
              5: "year",
              6: "old",
              7: "with",
              8: "0",
              9: "5",
            };
            return ids.map((id) => pieces[id] ?? "").join("");
          },
        };
        const pipe = new AutomaticSpeechRecognitionPipeline({
          task: PIPELINE_ID,
          model,
          tokenizer,
          processor,
        });

        const output = await pipe(new Float32Array(Math.ceil(20.1 * 16000)), {
          return_timestamps: "word",
          chunk_length_s: 20,
        });

        expect(output.text).toBe("score. 48-year-old with 0.5");
        expect(output.chunks).toEqual([
          { text: "score.", timestamp: [0, 0.4] },
          { text: "48-year-old", timestamp: [0.5, 1.3] },
          { text: "with", timestamp: [1.4, 1.7] },
          { text: "0.5", timestamp: [1.8, 2.05] },
        ]);
        expect(calls.map((x) => x.timeOffset)).toEqual([0, 10]);
      });

      it("auto-windows long Nemo audio with 90s sentence windows", async () => {
        const calls = [];
        const outputsByOffset = new Map([
          [0, {
            text: "Alpha. Beta. Gamma. Carry",
            words: [
              { text: "Alpha.", startTime: 0, endTime: 1 },
              { text: "Beta.", startTime: 30, endTime: 31 },
              { text: "Gamma.", startTime: 69, endTime: 70 },
              { text: "Carry", startTime: 84, endTime: 85 },
            ],
          }],
          [84, {
            text: "Carry on. Delta. Epsilon. Tail",
            words: [
              { text: "Carry", startTime: 84, endTime: 85 },
              { text: "on.", startTime: 86, endTime: 87 },
              { text: "Delta.", startTime: 110, endTime: 111 },
              { text: "Epsilon.", startTime: 139, endTime: 140 },
              { text: "Tail", startTime: 154, endTime: 155 },
            ],
          }],
          [154, {
            text: "Tail resumes. Zeta. Eta. Final",
            words: [
              { text: "Tail", startTime: 154, endTime: 155 },
              { text: "resumes.", startTime: 156, endTime: 157 },
              { text: "Zeta.", startTime: 180, endTime: 181 },
              { text: "Eta.", startTime: 209, endTime: 210 },
              { text: "Final", startTime: 224, endTime: 225 },
            ],
          }],
          [224, {
            text: "Final line. Omega.",
            words: [
              { text: "Final", startTime: 224, endTime: 225 },
              { text: "line.", startTime: 226, endTime: 227 },
              { text: "Omega.", startTime: 250, endTime: 251 },
            ],
          }],
        ]);
        const model = {
          config: { model_type: "nemo-conformer-tdt" },
          async transcribe(_inputs, options) {
            calls.push(options);
            const item = outputsByOffset.get(options.timeOffset);
            if (!item) {
              throw new Error(`Unexpected timeOffset ${options.timeOffset}`);
            }
            return {
              text: item.text,
              utteranceTimestamp: [item.words[0].startTime, item.words[item.words.length - 1].endTime],
              words: item.words,
            };
          },
          async dispose() {},
        };
        const processor = Object.assign(async () => ({ input_features: {} }), {
          feature_extractor: { config: { sampling_rate: 16000 } },
        });
        const tokenizer = {
          decode(ids) {
            const pieces = {
              1: "alpha",
              2: "beta",
              3: "gamma",
              4: "delta",
            };
            return ids.map((id) => pieces[id] ?? "").filter(Boolean).join(" ");
          },
        };
        const pipe = new AutomaticSpeechRecognitionPipeline({
          task: PIPELINE_ID,
          model,
          tokenizer,
          processor,
        });

        const output = await pipe(new Float32Array(300 * 16000), { return_timestamps: "word" });

        expect(output).toEqual({
          text: "Alpha. Beta. Gamma. Carry on. Delta. Epsilon. Tail resumes. Zeta. Eta. Final line. Omega.",
          chunks: [
            { text: "Alpha.", timestamp: [0, 1] },
            { text: "Beta.", timestamp: [30, 31] },
            { text: "Gamma.", timestamp: [69, 70] },
            { text: "Carry", timestamp: [84, 85] },
            { text: "on.", timestamp: [86, 87] },
            { text: "Delta.", timestamp: [110, 111] },
            { text: "Epsilon.", timestamp: [139, 140] },
            { text: "Tail", timestamp: [154, 155] },
            { text: "resumes.", timestamp: [156, 157] },
            { text: "Zeta.", timestamp: [180, 181] },
            { text: "Eta.", timestamp: [209, 210] },
            { text: "Final", timestamp: [224, 225] },
            { text: "line.", timestamp: [226, 227] },
            { text: "Omega.", timestamp: [250, 251] },
          ],
        });
        expect(calls).toHaveLength(4);
        expect(calls.map((x) => x.timeOffset)).toEqual([0, 84, 154, 224]);
        for (const call of calls) {
          expect(call).toMatchObject({
            returnTimestamps: true,
            returnWords: true,
            returnMetrics: false,
          });
        }
      });

      it("returns sentence chunks for auto-windowed long Nemo audio", async () => {
        const calls = [];
        const outputsByOffset = new Map([
          [0, {
            text: "Alpha. Beta. Gamma. Carry",
            words: [
              { text: "Alpha.", startTime: 0, endTime: 1 },
              { text: "Beta.", startTime: 30, endTime: 31 },
              { text: "Gamma.", startTime: 69, endTime: 70 },
              { text: "Carry", startTime: 84, endTime: 85 },
            ],
          }],
          [84, {
            text: "Carry on. Delta. Epsilon. Tail",
            words: [
              { text: "Carry", startTime: 84, endTime: 85 },
              { text: "on.", startTime: 86, endTime: 87 },
              { text: "Delta.", startTime: 110, endTime: 111 },
              { text: "Epsilon.", startTime: 139, endTime: 140 },
              { text: "Tail", startTime: 154, endTime: 155 },
            ],
          }],
          [154, {
            text: "Tail resumes. Zeta. Eta. Final",
            words: [
              { text: "Tail", startTime: 154, endTime: 155 },
              { text: "resumes.", startTime: 156, endTime: 157 },
              { text: "Zeta.", startTime: 180, endTime: 181 },
              { text: "Eta.", startTime: 209, endTime: 210 },
              { text: "Final", startTime: 224, endTime: 225 },
            ],
          }],
          [224, {
            text: "Final line. Omega.",
            words: [
              { text: "Final", startTime: 224, endTime: 225 },
              { text: "line.", startTime: 226, endTime: 227 },
              { text: "Omega.", startTime: 250, endTime: 251 },
            ],
          }],
        ]);
        const model = {
          config: { model_type: "nemo-conformer-tdt" },
          async transcribe(_inputs, options) {
            calls.push(options);
            const item = outputsByOffset.get(options.timeOffset);
            if (!item) {
              throw new Error(`Unexpected timeOffset ${options.timeOffset}`);
            }
            return {
              text: item.text,
              utteranceTimestamp: [item.words[0].startTime, item.words[item.words.length - 1].endTime],
              words: item.words,
            };
          },
          async dispose() {},
        };
        const processor = Object.assign(async () => ({ input_features: {} }), {
          feature_extractor: { config: { sampling_rate: 16000 } },
        });
        const pipe = new AutomaticSpeechRecognitionPipeline({
          task: PIPELINE_ID,
          model,
          tokenizer: {},
          processor,
        });

        const output = await pipe(new Float32Array(300 * 16000), { return_timestamps: true });

        expect(output).toEqual({
          text: "Alpha. Beta. Gamma. Carry on. Delta. Epsilon. Tail resumes. Zeta. Eta. Final line. Omega.",
          chunks: [
            { text: "Alpha.", timestamp: [0, 1] },
            { text: "Beta.", timestamp: [30, 31] },
            { text: "Gamma.", timestamp: [69, 70] },
            { text: "Carry on.", timestamp: [84, 87] },
            { text: "Delta.", timestamp: [110, 111] },
            { text: "Epsilon.", timestamp: [139, 140] },
            { text: "Tail resumes.", timestamp: [154, 157] },
            { text: "Zeta.", timestamp: [180, 181] },
            { text: "Eta.", timestamp: [209, 210] },
            { text: "Final line.", timestamp: [224, 227] },
            { text: "Omega.", timestamp: [250, 251] },
          ],
        });
        expect(calls.map((x) => x.timeOffset)).toEqual([0, 84, 154, 224]);
      });

      it("rejects non-finite audio samples before Nemo decoding", async () => {
        const { pipe } = makeUnitPipe();
        await expect(pipe(Float32Array.from([0, Number.NaN, 0]), { return_timestamps: false })).rejects.toThrow(
          "finite audio samples",
        );
      });

      it("disposes processor tensors after Nemo transcription when feature cache is disabled", async () => {
        let disposeCalls = 0;
        const model = {
          config: { model_type: "nemo-conformer-tdt" },
          async transcribe() {
            return { text: "ok" };
          },
          async dispose() {},
        };
        const processor = Object.assign(async () => {
          const input_features = new Tensor("float32", new Float32Array([0, 0]), [1, 1, 2]);
          const attention_mask = new Tensor("int64", BigInt64Array.from([1n]), [1, 1]);
          const trackDispose = (tensor) => {
            const originalDispose = tensor.dispose.bind(tensor);
            tensor.dispose = () => {
              disposeCalls += 1;
              originalDispose();
            };
          };
          trackDispose(input_features);
          trackDispose(attention_mask);
          return withNemoTensorOwnership({ input_features, attention_mask }, false);
        }, {
          feature_extractor: { config: { sampling_rate: 16000 } },
        });
        const pipe = new AutomaticSpeechRecognitionPipeline({
          task: PIPELINE_ID,
          model,
          tokenizer: {},
          processor,
        });

        const output = await pipe(new Float32Array(16000), { return_timestamps: false });
        expect(output).toEqual({ text: "ok" });
        expect(disposeCalls).toBe(2);
      });

      it("keeps processor tensors alive when Nemo feature cache owns tensor lifetimes", async () => {
        let disposeCalls = 0;
        let lastInputs = null;
        const model = {
          config: { model_type: "nemo-conformer-tdt" },
          async transcribe() {
            return { text: "ok" };
          },
          async dispose() {},
        };
        const processor = Object.assign(async () => {
          const input_features = new Tensor("float32", new Float32Array([0, 0]), [1, 1, 2]);
          const attention_mask = new Tensor("int64", BigInt64Array.from([1n]), [1, 1]);
          const trackDispose = (tensor) => {
            const originalDispose = tensor.dispose.bind(tensor);
            tensor.dispose = () => {
              disposeCalls += 1;
              originalDispose();
            };
          };
          trackDispose(input_features);
          trackDispose(attention_mask);
          lastInputs = withNemoTensorOwnership({ input_features, attention_mask }, true);
          return lastInputs;
        }, {
          feature_extractor: { config: { sampling_rate: 16000 } },
        });
        const pipe = new AutomaticSpeechRecognitionPipeline({
          task: PIPELINE_ID,
          model,
          tokenizer: {},
          processor,
        });

        try {
          const output = await pipe(new Float32Array(16000), { return_timestamps: false });
          expect(output).toEqual({ text: "ok" });
          expect(disposeCalls).toBe(0);
        } finally {
          lastInputs?.input_features.dispose();
          lastInputs?.attention_mask.dispose();
        }
      });

      it("disposes processor tensors when Nemo feature cache limits disable caching", async () => {
        let disposeCalls = 0;
        const model = {
          config: { model_type: "nemo-conformer-tdt" },
          async transcribe() {
            return { text: "ok" };
          },
          async dispose() {},
        };
        const processor = Object.assign(async () => {
          const input_features = new Tensor("float32", new Float32Array([0, 0]), [1, 1, 2]);
          const attention_mask = new Tensor("int64", BigInt64Array.from([1n]), [1, 1]);
          const trackDispose = (tensor) => {
            const originalDispose = tensor.dispose.bind(tensor);
            tensor.dispose = () => {
              disposeCalls += 1;
              originalDispose();
            };
          };
          trackDispose(input_features);
          trackDispose(attention_mask);
          return withNemoTensorOwnership({ input_features, attention_mask }, false);
        }, {
          feature_extractor: { config: { sampling_rate: 16000 } },
        });
        const pipe = new AutomaticSpeechRecognitionPipeline({
          task: PIPELINE_ID,
          model,
          tokenizer: {},
          processor,
        });

        const output = await pipe(new Float32Array(16000), { return_timestamps: false });
        expect(output).toEqual({ text: "ok" });
        expect(disposeCalls).toBe(2);
      });
    });
  });
};
