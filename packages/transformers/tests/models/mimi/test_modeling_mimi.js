import { EncodecFeatureExtractor, MimiModel, MimiEncoderModel, MimiDecoderModel } from "../../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("MimiModel", () => {
    const model_id = "hf-internal-testing/tiny-random-MimiModel";

    /** @type {MimiModel} */
    let model;
    /** @type {EncodecFeatureExtractor} */
    let feature_extractor;
    let inputs;
    beforeAll(async () => {
      model = await MimiModel.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      feature_extractor = await EncodecFeatureExtractor.from_pretrained(model_id);
      inputs = await feature_extractor(new Float32Array(12000));
    }, MAX_MODEL_LOAD_TIME);

    it(
      "forward",
      async () => {
        const { audio_values } = await model(inputs);
        expect(audio_values.dims).toEqual([1, 1, 13440]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "encode & decode",
      async () => {
        const encoder_outputs = await model.encode(inputs);
        expect(encoder_outputs.audio_codes.dims).toEqual([1, model.config.num_quantizers, 7]);

        const { audio_values } = await model.decode(encoder_outputs);
        expect(audio_values.dims).toEqual([1, 1, 13440]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe("MimiEncoderModel and MimiDecoderModel", () => {
    const model_id = "hf-internal-testing/tiny-random-MimiModel";

    /** @type {MimiEncoderModel} */
    let encoder_model;
    /** @type {MimiDecoderModel} */
    let decoder_model;
    /** @type {EncodecFeatureExtractor} */
    let feature_extractor;
    let inputs;
    let encoder_outputs;

    beforeAll(async () => {
      encoder_model = await MimiEncoderModel.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      decoder_model = await MimiDecoderModel.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      feature_extractor = await EncodecFeatureExtractor.from_pretrained(model_id);
      inputs = await feature_extractor(new Float32Array(12000));
    }, MAX_MODEL_LOAD_TIME);

    it(
      "encode",
      async () => {
        encoder_outputs = await encoder_model(inputs);
        expect(encoder_outputs.audio_codes.dims).toEqual([1, encoder_model.config.num_quantizers, 7]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "decode",
      async () => {
        const { audio_values } = await decoder_model(encoder_outputs);
        expect(audio_values.dims).toEqual([1, 1, 13440]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await encoder_model?.dispose();
      await decoder_model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
