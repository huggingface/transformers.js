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
        await pipe.dispose();
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
            const target = { text: "<unk>K" };
            expect(output).toEqual(target);
          },
          MAX_TEST_EXECUTION_TIME,
        );
      });

      afterAll(async () => {
        await pipe.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });
  });
};
