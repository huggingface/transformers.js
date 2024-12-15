import { pipeline, ZeroShotImageClassificationPipeline } from "../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";
import { load_cached_image } from "../asset_cache.js";

const PIPELINE_ID = "zero-shot-image-classification";

export default () => {
  describe("Zero-shot Image Classification", () => {
    const model_id = "hf-internal-testing/tiny-random-GroupViTModel";

    // Example adapted from https://huggingface.co/docs/transformers/en/model_doc/groupvit
    const labels = ["cat", "dog"];
    const hypothesis_template = "a photo of a {}";

    /** @type {ZeroShotImageClassificationPipeline} */
    let pipe;
    let images;
    beforeAll(async () => {
      pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
      images = await Promise.all([load_cached_image("white_image"), load_cached_image("blue_image")]);
    }, MAX_MODEL_LOAD_TIME);

    it("should be an instance of ZeroShotImageClassificationPipeline", () => {
      expect(pipe).toBeInstanceOf(ZeroShotImageClassificationPipeline);
    });

    describe("batch_size=1", () => {
      it(
        "default",
        async () => {
          const output = await pipe(images[0], labels);
          const target = [
            { score: 0.5990662574768066, label: "cat" },
            { score: 0.40093377232551575, label: "dog" },
          ];
          expect(output).toBeCloseToNested(target, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "custom (w/ hypothesis_template)",
        async () => {
          const output = await pipe(images[0], labels, { hypothesis_template });
          const target = [
            { score: 0.5527022480964661, label: "cat" },
            { score: 0.44729775190353394, label: "dog" },
          ];
          expect(output).toBeCloseToNested(target, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    describe("batch_size>1", () => {
      it(
        "default",
        async () => {
          const output = await pipe(images, labels);
          const target = [
            [
              { score: 0.5990662574768066, label: "cat" },
              { score: 0.40093377232551575, label: "dog" },
            ],
            [
              { score: 0.5006340146064758, label: "dog" },
              { score: 0.49936598539352417, label: "cat" },
            ],
          ];
          expect(output).toBeCloseToNested(target, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "custom (w/ hypothesis_template)",
        async () => {
          const output = await pipe(images, labels, { hypothesis_template });
          const target = [
            [
              { score: 0.5527022480964661, label: "cat" },
              { score: 0.44729775190353394, label: "dog" },
            ],
            [
              { score: 0.5395973324775696, label: "cat" },
              { score: 0.46040263772010803, label: "dog" },
            ],
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
