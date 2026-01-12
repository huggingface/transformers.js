import { AutoImageProcessor, rand, Tensor, VitPoseImageProcessor } from "../../../src/transformers.js";

import { load_cached_image } from "../../asset_cache.js";
import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  describe("VitPoseImageProcessor", () => {
    const model_id = "onnx-community/vitpose-base-simple";

    /** @type {VitPoseImageProcessor} */
    let processor;
    beforeAll(async () => {
      processor = await AutoImageProcessor.from_pretrained(model_id);
    }, MAX_PROCESSOR_LOAD_TIME);

    it(
      "default",
      async () => {
        const image = await load_cached_image("tiger");
        const { pixel_values, original_sizes, reshaped_input_sizes } = await processor(image);

        expect(pixel_values.dims).toEqual([1, 3, 256, 192]);
        expect(pixel_values.mean().item()).toBeCloseTo(-0.2771204710006714, 6);

        expect(original_sizes).toEqual([[408, 612]]);
        expect(reshaped_input_sizes).toEqual([[256, 192]]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "post_process_pose_estimation",
      async () => {
        const num_classes = 17;
        const size = [0, 0, 1000, 1500];
        const heatmaps = rand([1, num_classes, 64, 48]);

        const boxes = [[size]];
        const { bbox, scores, labels, keypoints } = processor.post_process_pose_estimation(heatmaps, boxes, { threshold: null })[0][0];

        expect(bbox).toEqual(size);
        expect(scores).toHaveLength(num_classes);
        expect(labels).toHaveLength(num_classes);
        expect(keypoints).toHaveLength(num_classes);
        expect(keypoints[0]).toHaveLength(2);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
