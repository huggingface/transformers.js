import { pipeline, ZeroShotObjectDetectionPipeline } from "../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";
import { load_cached_image } from "../asset_cache.js";

const PIPELINE_ID = "zero-shot-object-detection";

export default () => {
  describe("Zero-shot Object Detection", () => {
    describe("w/ post_process_object_detection", () => {
      const model_id = "hf-internal-testing/tiny-random-OwlViTForObjectDetection";

      const candidate_labels = ["hello", "hello world"];

      /** @type {ZeroShotObjectDetectionPipeline} */
      let pipe;
      let images;
      beforeAll(async () => {
        pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
        images = await Promise.all([load_cached_image("white_image"), load_cached_image("blue_image")]);
      }, MAX_MODEL_LOAD_TIME);

      const targets = {
        white_image: [
          {
            score: 0.6028420329093933,
            label: "hello",
            box: { xmin: 47, ymin: 117, xmax: 62, ymax: 134 },
          },
          {
            score: 0.6026064157485962,
            label: "hello world",
            box: { xmin: 47, ymin: 117, xmax: 62, ymax: 134 },
          },
          {
            score: 0.5987668037414551,
            label: "hello world",
            box: { xmin: 145, ymin: 47, xmax: 160, ymax: 63 },
          },
          {
            score: 0.5986272692680359,
            label: "hello",
            box: { xmin: 89, ymin: 131, xmax: 104, ymax: 148 },
          },
          {
            score: 0.5985949039459229,
            label: "hello world",
            box: { xmin: 89, ymin: 131, xmax: 104, ymax: 148 },
          },
          // ... many more
        ],

        blue_image: [
          {
            score: 0.6622366309165955,
            label: "hello",
            box: { xmin: 48, ymin: 45, xmax: 62, ymax: 61 },
          },
          {
            score: 0.6562080383300781,
            label: "hello world",
            box: { xmin: 48, ymin: 45, xmax: 62, ymax: 61 },
          },
          {
            score: 0.6493991613388062,
            label: "hello world",
            box: { xmin: 34, ymin: 58, xmax: 48, ymax: 74 },
          },
          {
            score: 0.6476974487304688,
            label: "hello",
            box: { xmin: 34, ymin: 58, xmax: 48, ymax: 74 },
          },
          {
            score: 0.6391685009002686,
            label: "hello",
            box: { xmin: 103, ymin: 59, xmax: 117, ymax: 75 },
          },
          // ... many more
        ],
      };

      it("should be an instance of ZeroShotObjectDetectionPipeline", () => {
        expect(pipe).toBeInstanceOf(ZeroShotObjectDetectionPipeline);
      });

      describe("batch_size=1", () => {
        it(
          "default",
          async () => {
            const output = await pipe(images[0], candidate_labels);
            expect(output).toHaveLength(512);

            expect(output.slice(0, targets.white_image.length)).toBeCloseToNested(targets.white_image, 5);
          },
          MAX_TEST_EXECUTION_TIME,
        );
        it(
          "custom (w/ top_k & threshold)",
          async () => {
            const top_k = 3;
            const output = await pipe(images[0], candidate_labels, { top_k, threshold: 0.05 });
            expect(output).toBeCloseToNested(targets.white_image.slice(0, top_k), 5);
          },
          MAX_TEST_EXECUTION_TIME,
        );
      });

      describe("batch_size>1", () => {
        it(
          "default",
          async () => {
            const output = await pipe(images, candidate_labels);
            const target = Object.values(targets);
            expect(output.map((x, i) => x.slice(0, target[i].length))).toBeCloseToNested(target, 5);
          },
          MAX_TEST_EXECUTION_TIME,
        );
        it(
          "custom (w/ top_k & threshold)",
          async () => {
            const top_k = 3;
            const output = await pipe(images, candidate_labels, { top_k, threshold: 0.05 });
            const target = Object.values(targets).map((x) => x.slice(0, top_k));
            expect(output).toBeCloseToNested(target, 5);
          },
          MAX_TEST_EXECUTION_TIME,
        );
      });

      afterAll(async () => {
        await pipe.dispose();
      }, MAX_MODEL_DISPOSE_TIME);
    });

    describe("w/ post_process_grounded_object_detection", () => {
      const model_id = "hf-internal-testing/tiny-random-GroundingDinoForObjectDetection";

      const candidate_labels = ["a cat."];

      /** @type {ZeroShotObjectDetectionPipeline} */
      let pipe;
      let image;
      beforeAll(async () => {
        pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
        image = await load_cached_image("white_image");
      }, MAX_MODEL_LOAD_TIME);

      const target = [
        { box: { xmax: 112, xmin: -111, ymax: 0, ymin: 0 }, label: "a cat. [SEP]", score: 1 },
        { box: { xmax: 112, xmin: -111, ymax: 0, ymin: 0 }, label: "a cat. [SEP]", score: 1 },
        { box: { xmax: 112, xmin: -111, ymax: 0, ymin: 0 }, label: "a cat. [SEP]", score: 1 },
        // ... many more
      ];

      it("should be an instance of ZeroShotObjectDetectionPipeline", () => {
        expect(pipe).toBeInstanceOf(ZeroShotObjectDetectionPipeline);
      });

      describe("batch_size=1", () => {
        it(
          "default",
          async () => {
            const output = await pipe(image, candidate_labels);
            expect(output).toHaveLength(900);
            expect(output.slice(0, target.length)).toBeCloseToNested(target, 5);
          },
          MAX_TEST_EXECUTION_TIME,
        );
        it(
          "custom (w/ top_k & threshold)",
          async () => {
            const top_k = 3;
            const output = await pipe(image, candidate_labels, { top_k, threshold: 0.05 });
            expect(output).toBeCloseToNested(target.slice(0, top_k), 5);
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
