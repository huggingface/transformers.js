import { PatchTSTModel, PatchTSTForPrediction, Tensor } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  const dims = [64, 512, 7];
  const prod = dims.reduce((a, b) => a * b, 1);
  const past_values = new Tensor(
    "float32",
    Float32Array.from({ length: prod }, (_, i) => i / prod),
    dims,
  );

  describe("PatchTSTModel", () => {
    const model_id = "hf-internal-testing/tiny-random-PatchTSTModel";

    /** @type {PatchTSTModel} */
    let model;
    beforeAll(async () => {
      model = await PatchTSTModel.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "default",
      async () => {
        const { last_hidden_state } = await model({ past_values });

        const { num_input_channels, d_model } = model.config;
        expect(last_hidden_state.dims).toEqual([dims[0], num_input_channels, 43, d_model]);
        expect(last_hidden_state.mean().item()).toBeCloseTo(0.016672514379024506, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("PatchTSTForPrediction", () => {
    const model_id = "onnx-community/granite-timeseries-patchtst";

    /** @type {PatchTSTForPrediction} */
    let model;
    beforeAll(async () => {
      model = await PatchTSTForPrediction.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
    }, MAX_MODEL_LOAD_TIME);

    it(
      "default",
      async () => {
        const { prediction_outputs } = await model({ past_values });

        const { prediction_length, num_input_channels } = model.config;
        expect(prediction_outputs.dims).toEqual([dims[0], prediction_length, num_input_channels]);
        expect(prediction_outputs.mean().item()).toBeCloseTo(0.506528377532959, 5);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
