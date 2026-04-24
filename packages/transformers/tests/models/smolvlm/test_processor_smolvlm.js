import { AutoProcessor, RawImage, RawVideo } from "../../../src/transformers.js";

import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

export default () => {
    describe("SmolVLMProcessorTest", () => {
        const model_id = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct";

        let processor;

        const video1 = new RawVideo(new Array(8).fill(new RawImage(new Uint8ClampedArray(400 * 300 * 3), 400, 300, 3)), 8);
        const video2 = new RawVideo(new Array(3).fill(new RawImage(new Uint8ClampedArray(400 * 300 * 3), 400, 300, 3)), 5);

        beforeAll(async () => {
            processor = await AutoProcessor.from_pretrained(model_id);
        }, MAX_PROCESSOR_LOAD_TIME);

        it("processes chat messages with video: vision tensors per frame", async () => {
            const messages = [
                {
                    role: "user",
                    content: [
                        { type: "video", url: "tiny_video.mp4" },
                        { type: "text", text: "What is shown in this video?" },
                    ],
                },
            ];
            const out = await processor(messages, null, { videos: video1 });
            expect(out.pixel_values.dims).toEqual([1, 8, 3, 512, 512]);
            expect(out.pixel_attention_mask.dims).toEqual([1, 8, 512, 512]);
        }, MAX_TEST_EXECUTION_TIME);

        it("expands <video> with more frames into longer tokenized input", async () => {
            const eightFrames = await processor("<video>x", null, { videos: video1 });
            const threeFrames = await processor("<video>x", null, { videos: video2 });
            expect(eightFrames.input_ids.size).toBeGreaterThan(threeFrames.input_ids.size);
        }, MAX_TEST_EXECUTION_TIME);

        it("processes batched <video> strings (vision batch + tokenizer kwargs)", async () => {
            const inputs = await processor(["<video>a", "<video>b"], null, {
                videos: video1,
                padding: "max_length",
                max_length: 200,
                truncation: true,
                do_rescale: true,
            });
            expect(inputs.pixel_values.dims[0]).toBe(2);
            expect(inputs.pixel_values.dims[1]).toBe(8);
            expect(inputs.input_ids.dims[1]).toBeGreaterThan(200);
        }, MAX_TEST_EXECUTION_TIME);

        it("matches frame count when RawVideo has fewer frames", async () => {
            const out = await processor("<video>x", null, { videos: video2 });
            expect(out.pixel_values.dims[1]).toBe(3);
            expect(out.pixel_attention_mask.dims[1]).toBe(3);
        }, MAX_TEST_EXECUTION_TIME);
    });
};
