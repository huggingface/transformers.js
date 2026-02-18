import { MgpstrProcessor, MgpstrForSceneTextRecognition } from "../../../src/transformers.js";

import { load_cached_image } from "../../asset_cache.js";
import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../../init.js";

export default () => {
  describe("MgpstrForSceneTextRecognition", () => {
    const model_id = "onnx-community/tiny-random-MgpstrForSceneTextRecognition";
    /** @type {MgpstrForSceneTextRecognition} */
    let model;
    /** @type {MgpstrProcessor} */
    let processor;
    beforeAll(async () => {
      model = await MgpstrForSceneTextRecognition.from_pretrained(model_id, DEFAULT_MODEL_OPTIONS);
      processor = await MgpstrProcessor.from_pretrained(model_id);
    }, MAX_MODEL_LOAD_TIME);

    const TARGETS = {
      white_image: {
        generated_text: ["mmmmmmmmmmmmmmmmmmmmmmmmmm"],
        scores: [3.5553885547065065e-27],
        char_preds: ["mmmmmmmmmmmmmmmmmmmmmmmmmm"],
        bpe_preds: ["wwwwwwwwwwwwwwwwwwwwwwwwww"],
        wp_preds: ["[unused65][unused65][unused65][unused65][unused65][unused65][unused65][unused65][unused65][unused65][unused65][unused65][unused65][unused65][unused65][unused65][unused65][unused65][unused65][unused65][unused65][unused65][unused65][unused65][unused65][unused65]"],
      },
      blue_image: {
        generated_text: ["11111111111111111111111111"],
        scores: [9.739909092663214e-32],
        char_preds: ["11111111111111111111111111"],
        bpe_preds: ["22222222222222222222222222"],
        wp_preds: ["[unused59][unused59][unused59][unused59][unused59][unused59][unused59][unused59][unused59][unused59][unused59][unused59][unused59][unused59][unused59][unused59][unused59][unused59][unused59][unused59][unused59][unused59][unused59][unused59][unused59][unused59]"],
      },
    };

    it(
      "batch_size=1",
      async () => {
        const image_id = "white_image";
        const image = await load_cached_image(image_id);

        const inputs = await processor(image);
        const outputs = await model(inputs);

        const { max_token_length, num_character_labels, num_bpe_labels, num_wordpiece_labels } = model.config;
        expect(outputs.char_logits.dims).toEqual([1, /* 27 */ max_token_length, /* 38 */ num_character_labels]);
        expect(outputs.bpe_logits.dims).toEqual([1, /* 27 */ max_token_length, /* 99 */ num_bpe_labels]);
        expect(outputs.wp_logits.dims).toEqual([1, /* 27 */ max_token_length, /* 99 */ num_wordpiece_labels]);

        const decoded = processor.batch_decode(outputs.logits);
        expect(decoded).toBeCloseToNested(TARGETS[image_id]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "batch_size>1",
      async () => {
        const image_ids = ["white_image", "blue_image"];
        const images = await Promise.all(image_ids.map((image_id) => load_cached_image(image_id)));

        const inputs = await processor(images);
        const outputs = await model(inputs);

        const { max_token_length, num_character_labels, num_bpe_labels, num_wordpiece_labels } = model.config;
        expect(outputs.char_logits.dims).toEqual([images.length, /* 27 */ max_token_length, /* 38 */ num_character_labels]);
        expect(outputs.bpe_logits.dims).toEqual([images.length, /* 27 */ max_token_length, /* 99 */ num_bpe_labels]);
        expect(outputs.wp_logits.dims).toEqual([images.length, /* 27 */ max_token_length, /* 99 */ num_wordpiece_labels]);

        const decoded = processor.batch_decode(outputs.logits);
        const target = image_ids.reduce((acc, image_id) => {
          for (const key in TARGETS[image_id]) (acc[key] ??= []).push(...TARGETS[image_id][key]);
          return acc;
        }, {});

        expect(decoded).toBeCloseToNested(target);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    afterAll(async () => {
      await model?.dispose();
    }, MAX_MODEL_DISPOSE_TIME);
  });
};
