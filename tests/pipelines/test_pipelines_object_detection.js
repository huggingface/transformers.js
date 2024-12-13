import { pipeline, ObjectDetectionPipeline } from "../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";
import { load_cached_image } from "../asset_cache.js";

const PIPELINE_ID = "object-detection";

export default () => {
  describe("Object Detection", () => {
    describe("yolos", () => {
      const model_id = "Xenova/yolos-tiny";
      /** @type {ObjectDetectionPipeline} */
      let pipe;
      beforeAll(async () => {
        pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
      }, MAX_MODEL_LOAD_TIME);

      it("should be an instance of ObjectDetectionPipeline", () => {
        expect(pipe).toBeInstanceOf(ObjectDetectionPipeline);
      });

      it(
        "single + threshold",
        async () => {
          const image = await load_cached_image("cats");
          const output = await pipe(image, { threshold: 0.9 });

          const target = [
            {
              score: 0.9921281933784485,
              label: "remote",
              box: { xmin: 32, ymin: 78, xmax: 185, ymax: 117 },
            },
            {
              score: 0.9884883165359497,
              label: "remote",
              box: { xmin: 324, ymin: 82, xmax: 376, ymax: 191 },
            },
            {
              score: 0.9197800159454346,
              label: "cat",
              box: { xmin: 5, ymin: 56, xmax: 321, ymax: 469 },
            },
            {
              score: 0.9300552606582642,
              label: "cat",
              box: { xmin: 332, ymin: 25, xmax: 638, ymax: 369 },
            },
          ];
          expect(output).toBeCloseToNested(target, 5);
        },
        MAX_TEST_EXECUTION_TIME,
      );

      afterAll(async () => {
        await pipe.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });

    describe("tiny-random", () => {
      const model_id = "hf-internal-testing/tiny-random-DetrForObjectDetection";

      /** @type {ObjectDetectionPipeline} */
      let pipe;
      let images;

      beforeAll(async () => {
        pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
        images = await Promise.all([load_cached_image("white_image"), load_cached_image("blue_image")]);
      }, MAX_MODEL_LOAD_TIME);

      it("should be an instance of ObjectDetectionPipeline", () => {
        expect(pipe).toBeInstanceOf(ObjectDetectionPipeline);
      });

      describe("batch_size=1", () => {
        it(
          "default (threshold unset)",
          async () => {
            const output = await pipe(images[0]);
            const target = [];
            expect(output).toBeCloseToNested(target, 5);
          },
          MAX_TEST_EXECUTION_TIME,
        );
        it(
          "default (threshold=0)",
          async () => {
            const output = await pipe(images[0], { threshold: 0 });
            const target = [
              { score: 0.020360443741083145, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
              { score: 0.020360419526696205, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
              { score: 0.02036038413643837, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
              { score: 0.020360447466373444, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
              { score: 0.020360389724373817, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
              { score: 0.020360423251986504, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
              { score: 0.02036040835082531, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
              { score: 0.020360363647341728, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
              { score: 0.020360389724373817, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
              { score: 0.020360389724373817, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
              { score: 0.020360343158245087, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
              { score: 0.020360423251986504, label: "LABEL_31", box: { xmin: 56, ymin: 55, xmax: 169, ymax: 167 } },
            ];
            expect(output).toBeCloseToNested(target, 5);
          },
          MAX_TEST_EXECUTION_TIME,
        );
      });

      // TODO: Add batched support to object detection pipeline
      // describe('batch_size>1', () => {
      //     it('default (threshold unset)', async () => {
      //         const output = await pipe(images);
      //         console.log(output);
      //         const target = [];
      //         expect(output).toBeCloseToNested(target, 5);
      //     }, MAX_TEST_EXECUTION_TIME);
      //     it('default (threshold=0)', async () => {
      //         const output = await pipe(images, { threshold: 0 });
      //         console.log(output);
      //         const target = [];
      //         expect(output).toBeCloseToNested(target, 5);
      //     }, MAX_TEST_EXECUTION_TIME);
      // });

      afterAll(async () => {
        await pipe.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });
  });
};
