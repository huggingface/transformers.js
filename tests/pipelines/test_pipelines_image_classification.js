import { pipeline, ImageClassificationPipeline } from "../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";
import { load_cached_image } from "../asset_cache.js";

const PIPELINE_ID = "image-classification";

export default () => {
  describe("Image Classification", () => {
    const model_id = "hf-internal-testing/tiny-random-vit";
    /** @type {ImageClassificationPipeline} */
    let pipe;
    let images;
    beforeAll(async () => {
      pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
      images = await Promise.all([load_cached_image("white_image"), load_cached_image("blue_image")]);
    }, MAX_MODEL_LOAD_TIME);

    it("should be an instance of ImageClassificationPipeline", () => {
      expect(pipe).toBeInstanceOf(ImageClassificationPipeline);
    });

    describe("batch_size=1", () => {
      it(
        "default (top_k=5)",
        async () => {
          const output = await pipe(images[0]);
          const target = [
            { label: "LABEL_1", score: 0.5020533800125122 },
            { label: "LABEL_0", score: 0.4979466497898102 },
          ];
          expect(output).toBeCloseToNested(target, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "custom (top_k=1)",
        async () => {
          const output = await pipe(images[0], { top_k: 1 });
          const target = [{ label: "LABEL_1", score: 0.5020533800125122 }];
          expect(output).toBeCloseToNested(target, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
    });

    describe("batch_size>1", () => {
      it(
        "default (top_k=5)",
        async () => {
          const output = await pipe(images);
          const target = [
            [
              { label: "LABEL_1", score: 0.5020533800125122 },
              { label: "LABEL_0", score: 0.4979466497898102 },
            ],
            [
              { label: "LABEL_1", score: 0.519227921962738 },
              { label: "LABEL_0", score: 0.4807720482349396 },
            ],
          ];
          expect(output).toBeCloseToNested(target, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );
      it(
        "custom (top_k=1)",
        async () => {
          const output = await pipe(images, { top_k: 1 });
          const target = [[{ label: "LABEL_1", score: 0.5020533800125122 }], [{ label: "LABEL_1", score: 0.519227921962738 }]];
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
