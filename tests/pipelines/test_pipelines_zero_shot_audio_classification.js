import { pipeline, ZeroShotAudioClassificationPipeline } from "../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";
import { load_cached_audio } from "../asset_cache.js";

const PIPELINE_ID = "zero-shot-audio-classification";

export default () => {
  describe("Zero-shot Audio Classification", () => {
    const model_id = "hf-internal-testing/tiny-clap-htsat-unfused";

    const labels = ["cat", "dog"];
    const hypothesis_template = "sound of a {}";

    /** @type {ZeroShotAudioClassificationPipeline} */
    let pipe;
    let audio;
    beforeAll(async () => {
      pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
      audio = await load_cached_audio("mlk");
    }, MAX_MODEL_LOAD_TIME);

    it("should be an instance of ZeroShotAudioClassificationPipeline", () => {
      expect(pipe).toBeInstanceOf(ZeroShotAudioClassificationPipeline);
    });

    describe("batch_size=1", () => {
      it(
        "default",
        async () => {
          const output = await pipe(audio, labels);
          const target = [
            { score: 0.4990939795970917, label: "cat" },
            { score: 0.5009059906005859, label: "dog" },
          ];
          expect(output).toBeCloseToNested(target, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "custom (w/ hypothesis_template)",
        async () => {
          const output = await pipe(audio, labels, { hypothesis_template });
          const target = [
            { score: 0.4987950325012207, label: "cat" },
            { score: 0.5012049674987793, label: "dog" },
          ];
          expect(output).toBeCloseToNested(target, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    afterAll(async () => {
      await pipe.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
