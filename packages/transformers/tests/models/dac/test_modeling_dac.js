import { DacFeatureExtractor, DacModel, DacEncoderModel, DacDecoderModel } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("DacModel", () => {
    const model_id = "hf-internal-testing/tiny-random-DacModel";

    /** @type {DacModel} */
    let model;
    /** @type {DacFeatureExtractor} */
    let feature_extractor;
    let inputs;
    beforeAll(async () => {
      model = await DacModel.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      feature_extractor = await DacFeatureExtractor.from_pretrained(model_id);
      inputs = await feature_extractor(new Float32Array(12000));
    }, MAX_MODEL_LOAD_TIME);

    it(
      "forward",
      async () => {
        const { audio_values } = await model(inputs);
        expect(audio_values.dims).toEqual([1, 1, 11832]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "encode & decode",
      async () => {
        const encoder_outputs = await model.encode(inputs);
        expect(encoder_outputs.audio_codes.dims).toEqual([1, model.config.n_codebooks, 37]);

        const { audio_values } = await model.decode(encoder_outputs);
        expect(audio_values.dims).toEqual([1, 1, 11832]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("DacEncoderModel and DacDecoderModel", () => {
    const model_id = "hf-internal-testing/tiny-random-DacModel";

    /** @type {DacEncoderModel} */
    let encoder_model;
    /** @type {DacDecoderModel} */
    let decoder_model;
    /** @type {DacFeatureExtractor} */
    let feature_extractor;
    let inputs;
    let encoder_outputs;

    beforeAll(async () => {
      encoder_model = await DacEncoderModel.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      decoder_model = await DacDecoderModel.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      feature_extractor = await DacFeatureExtractor.from_pretrained(model_id);
      inputs = await feature_extractor(new Float32Array(12000));
    }, MAX_MODEL_LOAD_TIME);

    it(
      "encode",
      async () => {
        encoder_outputs = await encoder_model(inputs);
        expect(encoder_outputs.audio_codes.dims).toEqual([1, encoder_model.config.n_codebooks, 37]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "decode",
      async () => {
        const { audio_values } = await decoder_model(encoder_outputs);
        expect(audio_values.dims).toEqual([1, 1, 11832]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await encoder_model?.dispose();
      await decoder_model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
