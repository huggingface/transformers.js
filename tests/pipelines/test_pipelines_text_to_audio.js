import { pipeline, TextToAudioPipeline } from "../../src/transformers.js";

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
      await pipe.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
