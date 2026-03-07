import { pipeline, AutomaticSpeechRecognitionPipeline, Tensor } from "../../src/transformers.js";

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
          return_timestamps: false,
          return_words: false,
          return_metrics: false,
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
          return_timestamps: true,
          return_words: true,
          return_metrics: false,
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
          return_timestamps: true,
          return_words: true,
          return_metrics: false,
        });
      });

      it("merges overlapping windows when Nemo chunking is enabled", async () => {
        const calls = [];
        const model = {
          config: { model_type: "nemo-conformer-tdt" },
          async transcribe(_inputs, options) {
            calls.push(options);
            if (options.timeOffset === 0) {
              return {
                text: "hello world again",
                words: [
                  { text: "hello", start_time: 0, end_time: 0.5 },
                  { text: "world", start_time: 0.5, end_time: 1.1 },
                  { text: "again", start_time: 1.2, end_time: 1.8 },
                ],
                tokens: [
                  { id: 1, token: "hello", raw_token: "hello", is_word_start: true, start_time: 0, end_time: 0.5 },
                  { id: 2, token: "world", raw_token: "world", is_word_start: true, start_time: 0.5, end_time: 1.1 },
                  { id: 3, token: "again", raw_token: "again", is_word_start: true, start_time: 1.2, end_time: 1.8 },
                ],
              };
            }
            return {
              text: "again today",
              words: [
                { text: "again", start_time: 1.2, end_time: 1.8 },
                { text: "today", start_time: 1.8, end_time: 2.4 },
              ],
              tokens: [
                { id: 3, token: "again", raw_token: "again", is_word_start: true, start_time: 1.2, end_time: 1.8 },
                { id: 4, token: "today", raw_token: "today", is_word_start: true, start_time: 1.8, end_time: 2.4 },
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

        const output = await pipe(new Float32Array(3 * 16000), {
          return_timestamps: "word",
          chunk_length_s: 2,
          stride_length_s: 0.5,
        });

        expect(output).toEqual({
          text: "hello world again today",
          chunks: [
            { text: "hello", timestamp: [0, 0.5] },
            { text: "world", timestamp: [0.5, 1.1] },
            { text: "again", timestamp: [1.2, 1.8] },
            { text: "today", timestamp: [1.8, 2.4] },
          ],
        });
        expect(calls).toHaveLength(2);
        expect(calls[0]).toMatchObject({
          return_timestamps: true,
          return_words: true,
          return_tokens: true,
          return_metrics: false,
          timeOffset: 0,
        });
        expect(calls[1]).toMatchObject({
          return_timestamps: true,
          return_words: true,
          return_tokens: true,
          return_metrics: false,
          timeOffset: 1,
        });
      });

      it("reconstructs windowed Nemo text from merged words when token decode drops spaces", async () => {
        const model = {
          config: { model_type: "nemo-conformer-tdt" },
          async transcribe(_inputs, options) {
            if (options.timeOffset === 0) {
              return {
                text: "score. 48-year-old",
                words: [
                  { text: "score.", start_time: 0, end_time: 0.4 },
                  { text: "48-year-old", start_time: 0.5, end_time: 1.3 },
                ],
                tokens: [
                  { id: 1, token: "score", raw_token: "▁score", is_word_start: true, start_time: 0, end_time: 0.3 },
                  { id: 2, token: ".", raw_token: ".", is_word_start: false, start_time: 0.3, end_time: 0.4 },
                  { id: 3, token: "48", raw_token: "48", is_word_start: false, start_time: 0.5, end_time: 0.8 },
                  { id: 4, token: "-", raw_token: "-", is_word_start: false, start_time: 0.8, end_time: 0.85 },
                  { id: 5, token: "year", raw_token: "year", is_word_start: false, start_time: 0.85, end_time: 1.05 },
                  { id: 4, token: "-", raw_token: "-", is_word_start: false, start_time: 1.05, end_time: 1.1 },
                  { id: 6, token: "old", raw_token: "old", is_word_start: false, start_time: 1.1, end_time: 1.3 },
                ],
              };
            }
            return {
              text: "with 0.5",
              words: [
                { text: "with", start_time: 1.4, end_time: 1.7 },
                { text: "0.5", start_time: 1.8, end_time: 2.05 },
              ],
              tokens: [
                { id: 7, token: "with", raw_token: "▁with", is_word_start: true, start_time: 1.4, end_time: 1.7 },
                { id: 8, token: "0", raw_token: "0", is_word_start: false, start_time: 1.8, end_time: 1.9 },
                { id: 2, token: ".", raw_token: ".", is_word_start: false, start_time: 1.9, end_time: 1.95 },
                { id: 9, token: "5", raw_token: "5", is_word_start: false, start_time: 1.95, end_time: 2.05 },
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

        const output = await pipe(new Float32Array(3 * 16000), {
          return_timestamps: "word",
          chunk_length_s: 2,
          stride_length_s: 0.5,
        });

        expect(output.text).toBe("score. 48-year-old with 0.5");
        expect(output.chunks).toEqual([
          { text: "score.", timestamp: [0, 0.4] },
          { text: "48-year-old", timestamp: [0.5, 1.3] },
          { text: "with", timestamp: [1.4, 1.7] },
          { text: "0.5", timestamp: [1.8, 2.05] },
        ]);
      });

      it("auto-window long Nemo audio with 90s chunks and 10s stride", async () => {
        const calls = [];
        const wordsByOffset = new Map([
          [0, { id: 1, text: "alpha", start: 0, end: 1 }],
          [70, { id: 2, text: "beta", start: 85, end: 86 }],
          [140, { id: 3, text: "gamma", start: 155, end: 156 }],
          [210, { id: 4, text: "delta", start: 225, end: 226 }],
        ]);
        const model = {
          config: { model_type: "nemo-conformer-tdt" },
          async transcribe(_inputs, options) {
            calls.push(options);
            const item = wordsByOffset.get(options.timeOffset);
            if (!item) {
              throw new Error(`Unexpected timeOffset ${options.timeOffset}`);
            }
            return {
              text: item.text,
              words: [
                { text: item.text, start_time: item.start, end_time: item.end },
              ],
              tokens: [
                {
                  id: item.id,
                  token: item.text,
                  raw_token: item.text,
                  is_word_start: true,
                  start_time: item.start,
                  end_time: item.end,
                },
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
          text: "alpha beta gamma delta",
          chunks: [
            { text: "alpha", timestamp: [0, 1] },
            { text: "beta", timestamp: [85, 86] },
            { text: "gamma", timestamp: [155, 156] },
            { text: "delta", timestamp: [225, 226] },
          ],
        });
        expect(calls).toHaveLength(4);
        expect(calls.map((x) => x.timeOffset)).toEqual([0, 70, 140, 210]);
        for (const call of calls) {
          expect(call).toMatchObject({
            return_timestamps: true,
            return_words: true,
            return_tokens: true,
            return_metrics: false,
          });
        }
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
          return { input_features, attention_mask };
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
          lastInputs = { input_features, attention_mask };
          return lastInputs;
        }, {
          feature_extractor: {
            config: { sampling_rate: 16000 },
            feature_cache: { max_entries: 2, max_size_mb: 8 },
          },
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
          return { input_features, attention_mask };
        }, {
          feature_extractor: {
            config: { sampling_rate: 16000 },
            feature_cache: { max_entries: 0, max_size_mb: 8 },
          },
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
