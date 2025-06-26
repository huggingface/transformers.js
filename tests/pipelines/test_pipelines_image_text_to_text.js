import { pipeline, ImageTextToTextPipeline } from "../../src/transformers.js";

import { MAX_MODEL_LOAD_TIME, MAX_TEST_EXECUTION_TIME, MAX_MODEL_DISPOSE_TIME, DEFAULT_MODEL_OPTIONS } from "../init.js";
import { load_cached_image } from "../asset_cache.js";

const PIPELINE_ID = "image-text-to-text";

export default () => {
    describe("Image Text to Text", () => {
        const model_id = "onnx-community/Qwen2-VL-2B-Instruct";
        //TODO: Looks like this model is too big and is triggering timeout. Use smaller model.
        /** @type {ImageTextToTextPipeline} */
        let pipe;
        let images;
        beforeAll(async () => {
            pipe = await pipeline(PIPELINE_ID, model_id, DEFAULT_MODEL_OPTIONS);
            images = await Promise.all([load_cached_image("white_image"), load_cached_image("blue_image")]);
            texts = ["What is the color of the image?", "What is the color of the image?"];
        }, MAX_MODEL_LOAD_TIME);

        it("should be an instance of ImageTextToTextPipeline", () => {
            expect(pipe).toBeInstanceOf(ImageTextToTextPipeline);
        });

        describe("batch_size=1", () => {
            it("default", async () => {
                const output = await pipe(images[0], texts[0]);
                const target = [{ generated_text: "" }]; //TODO: What should I put here? Will depend on the model...
                expect(output).toEqual(target);
            }, MAX_TEST_EXECUTION_TIME);
        });
    });
}