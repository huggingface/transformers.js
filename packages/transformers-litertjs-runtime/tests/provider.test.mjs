import assert from "node:assert/strict";
import test from "node:test";

import { RawImage, Tensor } from "@huggingface/transformers";
import { LiteRtInferenceProvider, postProcessYolo, preprocessYolo, resolveModelDescriptor, selectArtifact, visionTesting } from "../dist/transformers-litertjs-runtime.js";

test("resolves every supported benchmark model", () => {
  const expected = [
    ["litert-community/efficientnet_b0", "image-classification"],
    ["litert-community/inception_v3", "image-classification"],
    ["litert-community/resnet50", "image-classification"],
    ["litert-community/whisper-base", "automatic-speech-recognition"],
    ["litert-community/yolo26n", "object-detection"],
  ];
  for (const [modelId, task] of expected) {
    const provider = LiteRtInferenceProvider.from_modelId(modelId);
    assert.deepEqual(provider.capabilities.tasks, [task]);
  }
});

test("pins model artifacts and rejects unavailable dtypes", () => {
  const inception = resolveModelDescriptor("inception_v3");
  assert.deepEqual(selectArtifact(inception, null), {
    file: "inception_v3.tflite",
    revision: "8adf46014deda51d76ea19693140660c7610b65e",
    sha256: "0116bb2d6fa277da4de65b33c11a8fa870840531dbcc7bda0aeccbecd3299dc1",
  });
  assert.throws(() => selectArtifact(inception, "q8"), /does not provide dtype/);

  const whisper = resolveModelDescriptor("whisper_base");
  assert.equal(selectArtifact(whisper, "int8").file, "whisper_base_30s_i8.tflite");
});

test("supports artifact source and file overrides", () => {
  const provider = LiteRtInferenceProvider.from_modelId("yolo26n", {
    artifactModelId: "/models",
    modelFile: "custom.tflite",
  });
  assert.equal(provider.artifactModelId, "/models");
  assert.deepEqual(provider.listModelArtifacts({ dtype: "q8" }), ["config.json", "custom.tflite"]);
});

test("does not forward provider selection when delegating artifact cache operations", async () => {
  const provider = LiteRtInferenceProvider.from_modelId("resnet50");
  const assertSanitized = (options) => {
    assert.equal(options.inferenceBackend, undefined);
    assert.equal(options.inferenceProvider, undefined);
  };
  const getModelFileMetadata = async (_modelId, _file, options) => {
    assertSanitized(options);
    return { exists: true, size: 10, fromCache: true };
  };
  const deleteModelFile = async (_modelId, _file, options) => {
    assertSanitized(options);
    return true;
  };
  const options = {
    getModelFileMetadata,
    deleteModelFile,
    inferenceBackend: provider,
    inferenceProvider: provider,
  };
  await assert.doesNotReject(provider.getModelArtifactMetadata("resnet50.tflite", options));
  assert.equal(await provider.deleteModelArtifact("resnet50.tflite", options), true);
});

test("transposes NCHW data to NHWC", () => {
  const input = new Float32Array([1, 2, 3, 4, 10, 20, 30, 40]);
  assert.deepEqual(Array.from(visionTesting.nchwToNhwc(input, [1, 2, 2, 2])), [1, 10, 2, 20, 3, 30, 4, 40]);
});

test("filters and suppresses YOLO detections", () => {
  const candidates = 3;
  const values = new Float32Array(6 * candidates);
  values.set([320, 322, 100], 0 * candidates); // cx
  values.set([320, 322, 100], 1 * candidates); // cy
  values.set([100, 100, 40], 2 * candidates); // width
  values.set([100, 100, 40], 3 * candidates); // height
  values.set([0.9, 0.8, 0.1], 4 * candidates); // class 0
  values.set([0.1, 0.2, 0.95], 5 * candidates); // class 1
  const output = { predictions: new Tensor("float32", values, [1, 6, candidates]) };

  const [result] = postProcessYolo(output, 0.5, [[320, 640]]);

  assert.deepEqual(result.classes, [1, 0]);
  assert.equal(result.scores[0], values[5 * candidates + 2]);
  assert.deepEqual(result.boxes[0], [80, 40, 120, 60]);
});

test("letterboxes YOLO input and reverses the transform", async () => {
  const image = new RawImage(new Uint8ClampedArray(640 * 320 * 3).fill(255), 640, 320, 3);
  const inputs = await preprocessYolo([image]);
  assert.deepEqual(inputs.pixel_values.dims, [1, 3, 640, 640]);
  assert.deepEqual(inputs.image_transform, {
    scale: 1,
    padX: 0,
    padY: 160,
    originalWidth: 640,
    originalHeight: 320,
  });
  assert.equal(inputs.pixel_values.data[0], Math.fround(114 / 255));
  assert.equal(inputs.pixel_values.data[160 * 640], 1);

  const values = new Float32Array(5);
  values.set([320, 320, 100, 100, 0.9]);
  const output = {
    predictions: new Tensor("float32", values, [1, 5, 1]),
    image_transform: inputs.image_transform,
  };
  const [result] = postProcessYolo(output, 0.5, [[320, 640]]);
  assert.deepEqual(result.boxes[0], [270, 110, 370, 210]);

  const [percentageResult] = postProcessYolo(output, 0.5, null);
  assert.deepEqual(percentageResult.boxes[0], [270 / 640, 110 / 320, 370 / 640, 210 / 320]);
});
