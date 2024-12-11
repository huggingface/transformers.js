import { AutoProcessor, AutoModelForAudioFrameClassification } from "../../../src/transformers.js";

import { MAX_TEST_EXECUTION_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";
import { compare } from "../../test_utils.js";

export default () => {
  const models_to_test = ["onnx-community/pyannote-segmentation-3.0"];

  let audio;
  beforeAll(async () => {
    const url = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/mlk.npy";
    const buffer = await (await fetch(url)).arrayBuffer();
    audio = Float32Array.from(new Float64Array(buffer));
  });

  it(
    `PyAnnoteForAudioFrameClassification`,
    async () => {
      const model_id = models_to_test[0];

      // Load model and processor
      const model = await AutoModelForAudioFrameClassification.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      const processor = await AutoProcessor.from_pretrained(model_id);

      // Check processor config
      expect(processor.sampling_rate).toEqual(16000);

      // Preprocess audio
      const inputs = await processor(audio);

      // Run model with inputs
      const { logits } = await model(inputs);
      compare(logits.dims, [1, 767, 7]);
      compare(logits.mean().item(), -4.822614669799805, 6);

      const result = processor.post_process_speaker_diarization(logits, audio.length);
      const target = [
        [
          { id: 0, start: 0, end: 1.0512535626298245, confidence: 0.7898106738171984 },
          { id: 2, start: 1.0512535626298245, end: 2.373798367228636, confidence: 0.8923380609065887 },
          { id: 0, start: 2.373798367228636, end: 3.5776532534660155, confidence: 0.6920057005438546 },
          { id: 2, start: 3.5776532534660155, end: 4.578039708226655, confidence: 0.8169249580865657 },
          { id: 3, start: 4.578039708226655, end: 6.2396985652867, confidence: 0.6921662061495533 },
          { id: 2, start: 6.2396985652867, end: 8.664364040384521, confidence: 0.705263573835628 },
          { id: 0, start: 8.664364040384521, end: 10.071687358098641, confidence: 0.6650650397924295 },
          { id: 2, start: 10.071687358098641, end: 12.598087048934833, confidence: 0.8999033333468749 },
          { id: 0, start: 12.598087048934833, end: 13.005023911888312, confidence: 0.37838892004965197 },
        ],
      ];
      compare(result, target);

      await model.dispose();
    },
    MAX_TEST_EXECUTION_TIME,
  );
};
