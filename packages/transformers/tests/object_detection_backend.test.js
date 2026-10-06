import { jest } from "@jest/globals";

import { ObjectDetectionPipeline } from "../src/pipelines/object-detection.js";
import { RawImage } from "../src/utils/image.js";
import { Tensor } from "../src/utils/tensor.js";

describe("object detection backend postprocessing", () => {
  it("prefers model-owned postprocessing when provided", async () => {
    const image = new RawImage(new Uint8ClampedArray(2 * 3 * 3), 2, 3, 3);
    const processorFallback = jest.fn();
    const processor = Object.assign(
      jest.fn(async () => ({ pixel_values: new Tensor("float32", new Float32Array(12), [1, 3, 2, 2]) })),
      { image_processor: { post_process_object_detection: processorFallback } },
    );
    const postProcessObjectDetection = jest.fn(async (_output, threshold, imageSizes) => [{ boxes: [[0, 0, 2, 3]], scores: [0.75], classes: [1] }]);
    const model = Object.assign(
      jest.fn(async () => ({ predictions: new Tensor("float32", [0], [1]) })),
      {
        config: { id2label: { 1: "object" } },
        capabilities: { objectDetection: { version: 1, postprocess: "model" } },
        postProcessObjectDetection,
      },
    );
    const pipeline = new ObjectDetectionPipeline({ task: "object-detection", model, processor });

    await expect(pipeline._call(image, { threshold: 0.5 })).resolves.toEqual([{ score: 0.75, label: "object", box: { xmin: 0, ymin: 0, xmax: 2, ymax: 3 } }]);
    expect(processor).toHaveBeenCalledWith([image]);
    expect(postProcessObjectDetection).toHaveBeenCalledWith(expect.any(Object), 0.5, [[3, 2]]);
    expect(processorFallback).not.toHaveBeenCalled();
  });

  it("uses model-owned preprocessing when declared", async () => {
    const image = new RawImage(new Uint8ClampedArray(2 * 3 * 3), 2, 3, 3);
    const processor = jest.fn();
    const modelInputs = { pixel_values: new Tensor("float32", new Float32Array(12), [1, 3, 2, 2]) };
    const preprocessObjectDetection = jest.fn(async () => modelInputs);
    const model = Object.assign(
      jest.fn(async () => ({ predictions: new Tensor("float32", [0], [1]) })),
      {
        config: { id2label: { 0: "object" } },
        capabilities: { objectDetection: { version: 1, preprocess: "model", postprocess: "model" } },
        preprocessObjectDetection,
        postProcessObjectDetection: async () => [{ boxes: [], scores: [], classes: [] }],
      },
    );
    const pipeline = new ObjectDetectionPipeline({ task: "object-detection", model, processor });

    await pipeline._call(image);

    expect(preprocessObjectDetection).toHaveBeenCalledWith([image]);
    expect(model).toHaveBeenCalledWith(modelInputs);
    expect(processor).not.toHaveBeenCalled();
  });
});
