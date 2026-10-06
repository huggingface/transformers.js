import { AutoConfig, AutoModel, AutoProcessor, EmbeddingGemma2Model, EmbeddingGemma2Processor, ModelRegistry, RawVideo, mean_pooling, pipeline } from "../../../src/transformers.js";

import { load_cached_image, load_cached_audio } from "../../asset_cache.js";
import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

// Expected embeddings are from the PyTorch model (transformers), on the same inputs. With images and videos, PyTorch ran
// on the outputs of this processor, since the JS and Python image resizing differ slightly (see the processor tests).

/** A video panning across `image` (there is no test video): frame `i` is the 480x480 crop at x = 4i. */
const pan = async (image, num_frames) => new RawVideo(await Promise.all(Array.from({ length: num_frames }, (_, i) => image.crop([4 * i, 0, 4 * i + 479, 479]))), num_frames);

export default () => {
  const model_id = "onnx-internal-testing/tiny-random-EmbeddingGemma2Model";

  describe.skip("EmbeddingGemma2Model", () => {
    /** @type {EmbeddingGemma2Model} */
    let model;
    /** @type {EmbeddingGemma2Processor} */
    let processor;
    let cats, square, mlk;
    beforeAll(async () => {
      model = await AutoModel.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      processor = await AutoProcessor.from_pretrained(model_id);
      cats = await load_cached_image("cats"); // 640x480: 266 soft tokens
      square = await cats.crop([0, 0, 479, 479]); // 256 soft tokens
      mlk = await load_cached_audio("mlk"); // 13 s
    }, MAX_MODEL_LOAD_TIME);

    it("loads every encoder", () => {
      expect(model).toBeInstanceOf(EmbeddingGemma2Model);
      expect(Object.keys(model.sessions)).toEqual(["model", "vision_encoder", "audio_encoder"]);
    });

    it(
      "text",
      async () => {
        const inputs = await processor(["task: search result | query: Which planet is known as the Red Planet?", "title: none | text: Mars, known for its reddish appearance, is often referred to as the Red Planet."]);
        const { last_hidden_state, sentence_embedding } = await model(inputs);
        expect(last_hidden_state.dims).toEqual([2, 25, 24]);
        expect(sentence_embedding.tolist()).toBeCloseToNested(
          [
            [-0.29127, 0.24146, -0.4558, 0.2683, 0.22195, -0.02839, -0.05189, 0.21966, 0.12398, -0.20155, -0.06816, 0.18999, 0.10477, -0.22536, -0.17991, 0.04592, -0.35226, 0.04974, 0.01853, -0.09641, -0.03356, -0.31582, 0.03438, -0.21186],
            [-0.03765, -0.04078, -0.19725, -0.25596, -0.42791, -0.25321, 0.11024, 0.29025, -0.07728, -0.31429, -0.2872, 0.06117, -0.02532, -0.22794, -0.16951, 0.12912, -0.23385, 0.04275, 0.05434, -0.07153, 0.23542, -0.19115, 0.27001, -0.17447],
          ],
          4,
        );

        // `sentence_embedding` is the mean-pooled and normalized `last_hidden_state`
        const pooled = mean_pooling(last_hidden_state, inputs.attention_mask).normalize(2, -1);
        expect(pooled.tolist()).toBeCloseToNested(sentence_embedding.tolist(), 6);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "images",
      async () => {
        const inputs = await processor(null, [[cats], [square]]);
        const { sentence_embedding } = await model(inputs);
        expect(sentence_embedding.tolist()).toBeCloseToNested(
          [
            [0.00818, 0.2023, -0.18144, -0.27924, -0.51505, -0.29848, 0.04608, 0.29663, -0.27147, -0.15123, -0.05902, -0.11016, 0.09165, -0.18521, 0.15579, 0.30914, -0.05208, -0.06983, -0.10223, -0.05981, 0.08744, -0.22496, 0.20949, -0.07175],
            [0.00427, 0.19615, -0.18752, -0.29387, -0.52669, -0.31301, 0.06546, 0.29218, -0.26855, -0.14797, -0.06544, -0.08813, 0.07239, -0.18444, 0.1518, 0.29478, -0.0431, -0.07625, -0.08942, -0.06082, 0.08622, -0.21587, 0.20197, -0.07408],
          ],
          4,
        );
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "audio",
      async () => {
        const inputs = await processor(null, null, [mlk, mlk.subarray(0, 32000)]);
        const { sentence_embedding } = await model(inputs);
        expect(sentence_embedding.tolist()).toBeCloseToNested(
          [
            [0.26605, -0.44724, 0.31891, -0.24965, -0.14588, -0.06758, 0.08383, 0.03911, -0.06512, 0.20211, -0.03253, 0.1823, -0.22662, -0.26175, -0.32742, -0.10889, 0.21103, 0.20293, 0.26936, -0.07249, -0.099, 0.08507, -0.01231, -0.17813],
            [0.41008, -0.35571, 0.17208, -0.1629, -0.13817, -0.02466, 0.19851, 0.087, -0.08363, 0.21642, -0.02888, 0.19547, -0.20719, -0.22948, -0.30421, -0.14138, 0.22108, 0.15338, 0.2741, -0.11359, -0.28076, -0.06326, -0.04346, -0.19061],
          ],
          4,
        );
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "video",
      async () => {
        const inputs = await processor(null, null, null, await pan(cats, 3));
        const { sentence_embedding } = await model(inputs);
        expect(sentence_embedding.tolist()).toBeCloseToNested([[0.09366, 0.18715, -0.04702, -0.19496, -0.52165, -0.19943, 0.00065, 0.27462, -0.26257, -0.12024, -0.02032, -0.25977, 0.12977, -0.24617, 0.20659, 0.33431, -0.01669, -0.02641, -0.14844, -0.06567, 0.06374, -0.20875, 0.2301, -0.13263]], 4);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "text with every modality",
      async () => {
        const text = ["<|image|> A photo.", "Listen to <|audio|>, watch <|video|> and look at <|image|>."];
        const inputs = await processor(text, [[cats], [square]], [mlk.subarray(0, 32000)], [await pan(cats, 3)]);
        const { last_hidden_state, sentence_embedding } = await model(inputs);
        expect(last_hidden_state.dims).toEqual([2, 692, 24]);
        expect(sentence_embedding.tolist()).toBeCloseToNested(
          [
            [0.00521, 0.05026, 0.0027, -0.19729, -0.44216, -0.37534, -0.05304, 0.38988, -0.33007, -0.11071, -0.05471, -0.25489, 0.15398, -0.1651, 0.11224, 0.12509, -0.0005, -0.03669, -0.00575, -0.09309, 0.23222, -0.11676, 0.25911, -0.23281],
            [0.01093, -0.00365, 0.08914, -0.16685, -0.25521, -0.38977, -0.0906, 0.38689, -0.32723, -0.16594, -0.05809, -0.3369, 0.18533, -0.11828, 0.11371, 0.07342, -0.07228, -0.06932, 0.00969, -0.19004, 0.2919, -0.03599, 0.2452, -0.27952],
          ],
          4,
        );
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "placeholder and feature counts must match",
      async () => {
        const inputs = await processor(["<|image|> A photo."], cats);
        const { pixel_values, image_position_ids } = await processor(null, square); // 256 soft tokens instead of 266
        await expect(model({ ...inputs, pixel_values, image_position_ids })).rejects.toThrow("The number of image tokens (266) and image features (256) do not match.");
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "feature-extraction pipeline",
      async () => {
        const extractor = await pipeline("feature-extraction", model_id, DEFAULT_MODEL_OPTIONS);
        const texts = ["task: search result | query: Which planet is known as the Red Planet?", "title: none | text: Mars, known for its reddish appearance, is often referred to as the Red Planet."];
        const output = await extractor(texts, { pooling: "mean", normalize: true });
        expect(output.tolist()).toBeCloseToNested(
          [
            [-0.29127, 0.24146, -0.4558, 0.2683, 0.22195, -0.02839, -0.05189, 0.21966, 0.12398, -0.20155, -0.06816, 0.18999, 0.10477, -0.22536, -0.17991, 0.04592, -0.35226, 0.04974, 0.01853, -0.09641, -0.03356, -0.31582, 0.03438, -0.21186],
            [-0.03765, -0.04078, -0.19725, -0.25596, -0.42791, -0.25321, 0.11024, 0.29025, -0.07728, -0.31429, -0.2872, 0.06117, -0.02532, -0.22794, -0.16951, 0.12912, -0.23385, 0.04275, 0.05434, -0.07153, 0.23542, -0.19115, 0.27001, -0.17447],
          ],
          4,
        );
        await extractor.dispose();
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });

  describe.skip("EmbeddingGemma2Model (text-only)", () => {
    /** @type {import("../../../src/configs.js").PretrainedConfig} */
    let config;
    /** @type {EmbeddingGemma2Model} */
    let model;
    beforeAll(async () => {
      // As in Python (`vision_config=None, audio_config=None`)
      config = await AutoConfig.from_pretrained(model_id);
      config.vision_config = config.audio_config = null;
      model = await AutoModel.from_pretrained(model_id, { ...DEFAULT_MODEL_OPTIONS, config });
    }, MAX_MODEL_LOAD_TIME);

    it("only loads the text model", async () => {
      expect(Object.keys(model.sessions)).toEqual(["model"]);
      expect(await ModelRegistry.get_model_files(model_id, { config, dtype: "fp32" })).toEqual(["config.json", "onnx/model.onnx", "onnx/model.onnx_data"]);
    });

    it(
      "rejects media",
      async () => {
        const processor = await AutoProcessor.from_pretrained(model_id);
        await expect(model(await processor(null, await load_cached_image("cats")))).rejects.toThrow("Model does not have a vision_encoder session.");
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
