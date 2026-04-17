import { AutoProcessor, SmolVLMProcessor, RawImage, RawVideo } from "../../../src/transformers.js";

import { load_cached_image } from "../../asset_cache.js";
import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

function get_split_image_expected_tokens(processor, image_rows, image_cols, fake_image_token_id, image_token_id, image_seq_len, global_img_tokens_id) {
    let text_split_images = [];
    for (let n_h = 0; n_h < image_rows; ++n_h) {
        for (let n_w = 0; n_w < image_cols; ++n_w) {
            text_split_images.push(fake_image_token_id);
            text_split_images.push(...processor.tokenizer(`<row_${n_h + 1}_col_${n_w + 1}>`, { add_special_tokens: false }).input_ids.data);
            text_split_images.push(...new Array(image_seq_len).fill(image_token_id));
        }
        text_split_images.push(...processor.tokenizer("\n", { add_special_tokens: false }).input_ids.data);
    }
    // remove last newline
    const newline_len = processor.tokenizer("\n", { add_special_tokens: false }).input_ids.size;
    text_split_images.splice(text_split_images.length - newline_len, newline_len);
    
    // add double newline, as it gets its own token
    text_split_images.push(...processor.tokenizer("\n\n", { add_special_tokens: false }).input_ids.data);
    text_split_images.push(fake_image_token_id);
    text_split_images.push(...global_img_tokens_id);
    text_split_images.push(...new Array(image_seq_len).fill(image_token_id));
    text_split_images.push(fake_image_token_id);
    return text_split_images;
}

export default () => {
    describe("SmolVLMProcessorTest", () => {
        const model_id = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct";

        /** @type {SmolVLMProcessor} */
        let processor;
        
        // Mock images/videos for faster local execution rather than fetching
        const image1 = new RawImage(new Uint8ClampedArray(400 * 300 * 3), 400, 300, 3);
        const image2 = new RawImage(new Uint8ClampedArray(400 * 300 * 3), 400, 300, 3);
        const image3 = new RawImage(new Uint8ClampedArray(400 * 300 * 3), 400, 300, 3);
        const video1 = new RawVideo(new Array(8).fill(new RawImage(new Uint8ClampedArray(400 * 300 * 3), 400, 300, 3)), 8);
        
        let bos_token, image_token, video_token, fake_image_token, global_img_token;
        let bos_token_id, image_token_id, fake_image_token_id, global_img_tokens_id, padding_token_id, image_seq_len;

        beforeAll(async () => {
            processor = await AutoProcessor.from_pretrained(model_id);
            bos_token = processor.tokenizer.bos_token;
            image_token = processor.image_token;
            video_token = processor.video_token;
            fake_image_token = processor.fake_image_token;
            global_img_token = processor.global_img_token;

            bos_token_id = processor.tokenizer.convert_tokens_to_ids(bos_token);
            image_token_id = processor.tokenizer.convert_tokens_to_ids(image_token);
            fake_image_token_id = processor.tokenizer.convert_tokens_to_ids(fake_image_token);
            global_img_tokens_id = Array.from(processor.tokenizer(global_img_token, { add_special_tokens: false }).input_ids.data);
            padding_token_id = processor.tokenizer.pad_token_id;
            image_seq_len = processor.image_seq_len;
        }, MAX_PROCESSOR_LOAD_TIME);

        it("test_process_interleaved_images_prompts_no_image_splitting", async () => {
            const inputs = await processor(null, image1, { do_image_splitting: false });
            const image1_expected_size = [512, 512];
            expect(inputs.pixel_values.dims).toEqual([1, 1, 3, ...image1_expected_size]);
            expect(inputs.pixel_attention_mask.dims).toEqual([1, 1, ...image1_expected_size]);

            const image_str = "<image>";
            const text_str = "In this image, we see";
            let text = image_str + text_str;
            let inputs_with_text = await processor(text, image1, { do_image_splitting: false });

            const tokenized_sentence = processor.tokenizer(text_str, { add_special_tokens: false });
            let expected_input_ids = [fake_image_token_id].concat(global_img_tokens_id).concat(new Array(image_seq_len).fill(image_token_id)).concat([fake_image_token_id]).concat(Array.from(tokenized_sentence.input_ids.data));
            
            inputs_with_text = await processor(text, image1, { add_special_tokens: false, do_image_splitting: false });
            expect(Array.from(inputs_with_text.input_ids.data)).toEqual(expected_input_ids.map(BigInt));
            expect(Array.from(inputs_with_text.attention_mask.data)).toEqual(new Array(expected_input_ids.length).fill(1n));
            expect(inputs_with_text.pixel_values.dims).toEqual([1, 1, 3, ...image1_expected_size]);
            expect(inputs_with_text.pixel_attention_mask.dims).toEqual([1, 1, ...image1_expected_size]);

            // Batch processing
            const text_str_1 = "In this image, we see";
            const text_str_2 = "In this image, we see";
            const text_batch = [
                image_str + text_str_1,
                image_str + image_str + text_str_2,
            ];
            const images = [[image1], [image2, image3]];

            const inputs_batch = await processor(text_batch, images, { padding: true, add_special_tokens: false, do_image_splitting: false });

            expect(inputs_batch.pixel_values.dims).toEqual([2, 2, 3, 512, 512]);
            expect(inputs_batch.pixel_attention_mask.dims).toEqual([2, 2, 512, 512]);
        }, MAX_TEST_EXECUTION_TIME);

        it("test_process_interleaved_images_prompts_image_splitting", async () => {
            const inputs = await processor(null, image1, { do_image_splitting: true });
            expect(inputs.pixel_values.dims).toEqual([1, 13, 3, 512, 512]);
            expect(inputs.pixel_attention_mask.dims).toEqual([1, 13, 512, 512]);

            const image_str = "<image>";
            const text_str = "In this image, we see";
            const text = image_str + text_str;
            const inputs_with_text = await processor(text, image1, { add_special_tokens: false, do_image_splitting: true });

            const tokenized_sentence = processor.tokenizer(text_str, { add_special_tokens: false });
            const split_image1_tokens = get_split_image_expected_tokens(processor, 3, 4, fake_image_token_id, image_token_id, image_seq_len, global_img_tokens_id);
            const expected_input_ids_1 = split_image1_tokens.concat(Array.from(tokenized_sentence.input_ids.data));
            
            expect(Array.from(inputs_with_text.input_ids.data)).toEqual(expected_input_ids_1.map(BigInt));
            expect(Array.from(inputs_with_text.attention_mask.data)).toEqual(new Array(expected_input_ids_1.length).fill(1n));
            expect(inputs_with_text.pixel_values.dims).toEqual([1, 13, 3, 512, 512]);
            expect(inputs_with_text.pixel_attention_mask.dims).toEqual([1, 13, 512, 512]);

            const text_str_1 = "In this image, we see";
            const text_str_2 = "bla, bla";
            const text_batch = [
                image_str + text_str_1,
                text_str_2 + image_str + image_str,
            ];
            const images = [[image1], [image2, image3]];

            const inputs_batch = await processor(text_batch, images, { padding: true, add_special_tokens: false, do_image_splitting: true });

            expect(inputs_batch.pixel_values.dims).toEqual([2, 26, 3, 512, 512]);
            expect(inputs_batch.pixel_attention_mask.dims).toEqual([2, 26, 512, 512]);
        }, MAX_TEST_EXECUTION_TIME);

        it("test_add_special_tokens_processor", async () => {
            const image_str = "<image>";
            const text_str = "In this image, we see";
            const text = text_str + image_str;

            const inputs = await processor(text, image1, { add_special_tokens: false });
            const tokenized_sentence = processor.tokenizer(text_str, { add_special_tokens: false });
            const split_image1_tokens = get_split_image_expected_tokens(processor, 3, 4, fake_image_token_id, image_token_id, image_seq_len, global_img_tokens_id);
            const expected_input_ids = Array.from(tokenized_sentence.input_ids.data).concat(split_image1_tokens);
            expect(Array.from(inputs.input_ids.data)).toEqual(expected_input_ids.map(BigInt));

            const inputs2 = await processor(text, image1);
            expect(Array.from(inputs2.input_ids.data)).toEqual(expected_input_ids.map(BigInt));
        }, MAX_TEST_EXECUTION_TIME);

        it("test_non_nested_images_with_batched_text", async () => {
            const image_str = "<image>";
            const text_str_1 = "In this image, we see";
            const text_str_2 = "In this image, we see";

            const text = [
                image_str + text_str_1,
                image_str + image_str + text_str_2,
            ];
            const images = [[image1], [image2, image3]];

            const inputs = await processor(text, images, { padding: true, do_image_splitting: false });

            expect(inputs.pixel_values.dims).toEqual([2, 2, 3, 512, 512]);
            expect(inputs.pixel_attention_mask.dims).toEqual([2, 2, 512, 512]);
        }, MAX_TEST_EXECUTION_TIME);

        it("test_process_interleaved_images_prompts_image_error", async () => {
            const text1 = [
                "This is a test sentence.",
                "In this other sentence we try some good things",
            ];
            const images1 = [[image1], [image2]];
            await expect(processor(text1, images1, { padding: true })).rejects.toThrow();

            const images2 = [[image1], []];
            await expect(processor(text1, images2, { padding: true })).rejects.toThrow();

            const text2 = [
                "This is a test sentence.<image>",
                "In this other sentence we try some good things<image>",
            ];
            const images3 = [[image1], [image2, image3]];
            await expect(processor(text2, images3, { padding: true })).rejects.toThrow();

            const images4 = [[], [image2]];
            await expect(processor(text2, images4, { padding: true })).rejects.toThrow();

            const images5 = [image1, image2, image3];
            await expect(processor(text2, images5, { padding: true })).rejects.toThrow();

            const images6 = [image1];
            await expect(processor(text2, images6, { padding: true })).rejects.toThrow();
        }, MAX_TEST_EXECUTION_TIME);

        it("test_apply_chat_template", async () => {
            const messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What do these images show?"},
                        {"type": "image"},
                        {"type": "image"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "The first image shows the statue of Liberty in New York. The second image picture depicts Idefix, the dog of Obelix in Asterix and Obelix.",
                        }
                    ],
                },
                {"role": "user", "content": [{"type": "text", "text": "And who is that?"}]},
            ];
            const rendered = processor.apply_chat_template(messages, { add_generation_prompt: true });

            const expected_rendered = 
                "<|im_start|>User: What do these images show?<image><image><end_of_utterance>\n" +
                "Assistant: The first image shows the statue of Liberty in New York. The second image picture depicts Idefix, the dog of Obelix in Asterix and Obelix.<end_of_utterance>\n" +
                "User: And who is that?<end_of_utterance>\n" +
                "Assistant:";
            
            expect(rendered).toEqual(expected_rendered);
        });

        it("test_apply_chat_template_video_frame_sampling", async () => {
            const messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "url": "tiny_video.mp4"
                        },
                        {"type": "text", "text": "What is shown in this video?"},
                    ],
                },
            ];

            const out_dict_with_video = await processor(messages, null, { videos: video1 });
            expect("pixel_values" in out_dict_with_video).toBeTruthy();
            expect(out_dict_with_video.pixel_values.dims[0]).toEqual(1);
            // pixel_values.dims = [1, 8, 3, 512, 512] for video since our mock video has 8 frames
            expect(out_dict_with_video.pixel_values.dims[1]).toEqual(8);
        }, MAX_TEST_EXECUTION_TIME);

        it("test_unstructured_kwargs_batched", async () => {
            const input_str = ["<image>text", "<image>text"];
            const image_input = [[image1], [image1]];
            const inputs = await processor(input_str, image_input, {
                padding: "max_length",
                max_length: 200,
                truncation: true,
                size: { longest_edge: 300 },
                do_image_splitting: false,
            });

            expect(inputs.pixel_values.dims[3]).toEqual(300);
            expect(inputs.pixel_values.dims[4]).toEqual(300);
            expect(inputs.input_ids.dims[1]).toEqual(200);
        }, MAX_TEST_EXECUTION_TIME);

        it("test_unstructured_kwargs_batched_video", async () => {
            const input_str = ["<video>text", "<video>text"];
            const inputs = await processor(input_str, null, {
                videos: video1,
                padding: "max_length",
                max_length: 200,
                truncation: true,
                do_rescale: true,
            });

            expect(inputs.input_ids.dims[1]).toEqual(200);
        }, MAX_TEST_EXECUTION_TIME);

        it("test_text_only_inference", async () => {
            const text = "This is a simple text without images.";
            const inputs = await processor(text, null, { add_special_tokens: false });

            const tokenized_sentence = processor.tokenizer(text, { add_special_tokens: false });
            expect(Array.from(inputs.input_ids.data)).toEqual(Array.from(tokenized_sentence.input_ids.data));
            expect(Array.from(inputs.attention_mask.data)).toEqual(new Array(tokenized_sentence.input_ids.size).fill(1n));
            expect(inputs.pixel_values).toBeUndefined();
            expect(inputs.pixel_attention_mask).toBeUndefined();
        });

        it("test_missing_images_error", async () => {
            const text = "Let me show you this image: <image> What do you think?";
            await expect(processor(text)).rejects.toThrow();

            const texts = [
                "First text with <image> token.",
                "Second text <image> with token.",
            ];
            await expect(processor(texts)).rejects.toThrow();
        });

        it("test_special_mm_token_truncation", async () => {
            const input_str = ["<image> text", "<image> text"];
            const image_input = [[image1], [image1]];
            
            // Should work fine with enough length
            await processor(input_str, image_input, {
                truncation: false,
                padding: true,
                do_image_splitting: false,
            });

            // Should throw error if truncated and multimodal tokens are lost
            await expect(processor(input_str, image_input, {
                truncation: true,
                padding: true,
                max_length: 5, // very small to force truncation of <image> tokens
                do_image_splitting: false,
            })).rejects.toThrow(/truncated/);
        }, MAX_TEST_EXECUTION_TIME);
    });
};
