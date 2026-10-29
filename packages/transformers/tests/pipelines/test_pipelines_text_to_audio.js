import { jest } from "@jest/globals";
import { pipeline, TextToAudioPipeline, ModelRegistry, Tensor } from "../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";

const PIPELINE_ID = "text-to-audio";

export default () => {
  describe("Text to Audio", () => {
    const model_id = "Xenova/tiny-random-vits";

    /** @type {TextToAudioPipeline} */
    let pipe;
    beforeAll(async () => {
      pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
    }, MAX_MODEL_LOAD_TIME);

    it("should be an instance of TextToAudioPipeline", () => {
      expect(pipe).toBeInstanceOf(TextToAudioPipeline);
    });

    it(
      "default",
      async () => {
        const output = await pipe("hello");
        expect(output.audio).toHaveLength(6400);
        // NOTE: The mean value is not deterministic, so we just check the first few digits
        expect(output.audio.reduce((a, b) => a + b, 0) / output.audio.length).toBeCloseTo(-0.0125, 2);
        expect(output.sampling_rate).toEqual(16000);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await pipe?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  it.each(["vits", "musicgen", "supertonic"])("dispatches %s by model type when a processor is present", async (model_type) => {
    const pipe = new TextToAudioPipeline({ model: { config: { model_type } }, processor: {} });
    const method = model_type === "supertonic" ? "_call_supertonic" : "_call_text_to_waveform";
    const call = jest.spyOn(pipe, method).mockResolvedValue("audio");
    expect(await pipe("Hello.", {})).toBe("audio");
    expect(call).toHaveBeenCalledTimes(1);
  });

  describe("Text to Audio (SpeechT5)", () => {
    it(
      "includes the processor in the pipeline files",
      async () => {
        const files = await ModelRegistry.get_pipeline_files(PIPELINE_ID, "Xenova/speecht5_tts", DEFAULT_MODEL_OPTIONS);
        expect(files).toContain("preprocessor_config.json");
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it("returns the vocoder's 1D waveform as audio", async () => {
      const waveform = new Tensor(
        "float32",
        Float32Array.from({ length: 256 }, (_, i) => i / 256),
        [256],
      );
      const model = {
        config: { model_type: "speecht5" },
        generate_speech: async () => ({ waveform }),
      };
      const tokenizer = () => ({ input_ids: new Tensor("int64", new BigInt64Array([4n, 5n, 2n]), [1, 3]) });
      const processor = { feature_extractor: { config: { sampling_rate: 16000 } } };
      const pipe = new TextToAudioPipeline({ task: PIPELINE_ID, model, tokenizer, processor, vocoder: {} });

      const output = await pipe("hello", { speaker_embeddings: new Float32Array(512) });
      expect(output.audio).toEqual(waveform.data);
      expect(output.sampling_rate).toEqual(16000);
    });
  });
};
