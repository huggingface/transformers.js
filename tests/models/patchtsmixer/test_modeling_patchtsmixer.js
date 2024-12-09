import { PatchTSMixerModel, PatchTSMixerForPrediction, Tensor } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  const dims = [64, 512, 7];
  const prod = dims.reduce((a, b) => a * b, 1);
  const past_values = new Tensor(
    "float32",
    Float32Array.from({ length: prod }, (_, i) => i / prod),
    dims,
  );

  describe("PatchTSMixerModel", () => {
    const model_id = "hf-internal-testing/tiny-random-PatchTSMixerModel";

    /** @type {PatchTSMixerModel} */
    let model;
    beforeAll(async () => {
      model = await PatchTSMixerModel.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "default",
      async () => {
        const { last_hidden_state } = await model({ past_values });

        const { num_input_channels, num_patches, d_model } = model.config;
        expect(last_hidden_state.dims).toEqual([dims[0], num_input_channels, num_patches, d_model]);
        expect(last_hidden_state.mean().item()).toBeCloseTo(0.03344963490962982, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("PatchTSMixerForPrediction", () => {
    const model_id = "onnx-community/granite-timeseries-patchtsmixer";

    /** @type {PatchTSMixerForPrediction} */
    let model;
    beforeAll(async () => {
      model = await PatchTSMixerForPrediction.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "default",
      async () => {
        const { prediction_outputs } = await model({ past_values });

        const { prediction_length, num_input_channels } = model.config;
        expect(prediction_outputs.dims).toEqual([dims[0], prediction_length, num_input_channels]);
        expect(prediction_outputs.mean().item()).toBeCloseTo(0.5064773559570312, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
