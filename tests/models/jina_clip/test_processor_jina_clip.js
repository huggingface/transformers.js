import { AutoProcessor, JinaCLIPProcessor } from "../../../src/transformers.js";
import { load_cached_image } from "../../asset_cache.js";

import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
  describe("JinaCLIPProcessor", () => {
    const model_id = "jinaai/jina-clip-v2";

    /** @type {JinaCLIPProcessor} */
    let processor;
    beforeAll(async () => {
      processor = await AutoProcessor.from_pretrained(model_id);
    }, MAX_PROCESSOR_LOAD_TIME);

    it(
      "Image and text",
      async () => {
        // Prepare inputs
        const images = [await load_cached_image("white_image"), await load_cached_image("blue_image")];
        const sentences = [
          "غروب جميل على الشاطئ", // Arabic
          "海滩上美丽的日落", // Chinese
          "Un beau coucher de soleil sur la plage", // French
          "Ein wunderschöner Sonnenuntergang am Strand", // German
          "Ένα όμορφο ηλιοβασίλεμα πάνω από την παραλία", // Greek
          "समुद्र तट पर एक खूबसूरत सूर्यास्त", // Hindi
          "Un bellissimo tramonto sulla spiaggia", // Italian
          "浜辺に沈む美しい夕日", // Japanese
          "해변 위로 아름다운 일몰", // Korean
        ];

        // Encode text and images
        const { input_ids, attention_mask, pixel_values } = await processor(sentences, images, { padding: true, truncation: true });

        expect(input_ids.dims).toEqual([sentences.length, 19]);
        expect(attention_mask.dims).toEqual([sentences.length, 19]);
        expect(pixel_values.dims).toEqual([images.length, 3, 512, 512]);
        expect(pixel_values.mean().item()).toBeCloseTo(0.7857685685157776, 6);
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
