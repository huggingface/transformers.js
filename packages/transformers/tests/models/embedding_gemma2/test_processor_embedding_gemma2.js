import { AutoProcessor, EmbeddingGemma2Processor, RawVideo } from "../../../src/transformers.js";

import { load_cached_image, load_cached_audio } from "../../asset_cache.js";
import { MAX_PROCESSOR_LOAD_TIME, MAX_TEST_EXECUTION_TIME } from "../../init.js";

// Expected values are from the Python processor (transformers), on the same inputs. Pixel values only match to about
// 1e-4, since the JS and Python image resizing differ slightly.
const IMAGE_TOKEN = 258880;
const AUDIO_TOKEN = 258881;
const VIDEO_TOKEN = 258884;

/** Number of `token_id` tokens per sample. */
const count = (input_ids, token_id) => input_ids.tolist().map((ids) => ids.filter((id) => Number(id) === token_id).length);

/** A video panning across `image` (there is no test video): frame `i` is the 480x480 crop at x = 4i. */
const pan = async (image, num_frames) => new RawVideo(await Promise.all(Array.from({ length: num_frames }, (_, i) => image.crop([4 * i, 0, 4 * i + 479, 479]))), num_frames);

export default () => {
  describe.skip("EmbeddingGemma2Processor", () => {
    const model_id = "onnx-community/embeddinggemma-2-ONNX";

    /** @type {EmbeddingGemma2Processor} */
    let processor;
    let cats, square, mlk, mlk_2s;
    beforeAll(async () => {
      processor = await AutoProcessor.from_pretrained(model_id);
      cats = await load_cached_image("cats"); // 640x480: 266 soft tokens
      square = await cats.crop([0, 0, 479, 479]); // 256 soft tokens
      mlk = await load_cached_audio("mlk"); // 13 s
      mlk_2s = mlk.subarray(0, 32000);
    }, MAX_PROCESSOR_LOAD_TIME);

    it("is loaded by AutoProcessor", () => {
      expect(processor).toBeInstanceOf(EmbeddingGemma2Processor);
    });

    it(
      "text",
      async () => {
        const { input_ids, attention_mask } = await processor(["task: search result | query: Which planet is known as the Red Planet?", "title: none | text: Mars, known for its reddish appearance, is often referred to as the Red Planet."]);
        expect(input_ids.tolist()).toEqual([
          [2n, 8071n, 236787n, 3927n, 1354n, 1109n, 7609n, 236787n, 15311n, 13401n, 563n, 3224n, 618n, 506n, 4855n, 38342n, 236881n, 1n, 0n, 0n, 0n, 0n, 0n, 0n, 0n],
          [2n, 3250n, 236787n, 7293n, 1109n, 1816n, 236787n, 23156n, 236764n, 3224n, 573n, 1061n, 73865n, 10086n, 236764n, 563n, 3187n, 11081n, 531n, 618n, 506n, 4855n, 38342n, 236761n, 1n],
        ]);
        expect(attention_mask.sum(-1).tolist()).toEqual([18n, 25n]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "chat template",
      async () => {
        const messages = [
          { role: "system", content: "task: search result | query: " },
          { role: "user", content: [{ type: "image" }, { type: "text", text: "Which planet is this?" }, { type: "audio" }] },
        ];
        expect(processor.apply_chat_template(messages)).toEqual("task: search result | query: <|image|>Which planet is this?<|audio|>");
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "image without text",
      async () => {
        const { input_ids, pixel_values, image_position_ids, num_soft_tokens_per_image } = await processor(null, cats);
        expect(input_ids.dims).toEqual([1, 270]);
        expect(input_ids.slice(null, [0, 3]).tolist()).toEqual([[2n, 255999n, BigInt(IMAGE_TOKEN)]]);
        expect(input_ids.slice(null, [-2, null]).tolist()).toEqual([[258882n, 1n]]);
        expect(count(input_ids, IMAGE_TOKEN)).toEqual([266]);
        expect(num_soft_tokens_per_image).toEqual([266]);
        expect(pixel_values.dims).toEqual([1, 2520, 768]);
        expect(pixel_values.mean().item()).toBeCloseTo(953534.3690403197 / (2520 * 768), 3);
        expect(image_position_ids.dims).toEqual([1, 2520, 2]);
        expect(image_position_ids.sum().item()).toEqual(115857n);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "images, one per sample",
      async () => {
        const { input_ids, attention_mask, pixel_values } = await processor(null, [[cats], [square]]);
        expect(input_ids.dims).toEqual([2, 270]);
        expect(attention_mask.sum(-1).tolist()).toEqual([270n, 260n]);
        expect(count(input_ids, IMAGE_TOKEN)).toEqual([266, 256]);
        expect(pixel_values.dims).toEqual([2, 2520, 768]);
        expect(pixel_values.mean().item()).toBeCloseTo(1870414.8333861837 / (2 * 2520 * 768), 3);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "flat list of images is one sample",
      async () => {
        const { input_ids } = await processor(null, [cats, cats]);
        expect(input_ids.dims).toEqual([1, 539]);
        expect(count(input_ids, IMAGE_TOKEN)).toEqual([532]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "audio, one sample per clip",
      async () => {
        const { input_ids, input_features, input_features_mask } = await processor(null, null, [mlk, mlk_2s]);
        const block = (n) => [256000n, ...Array(n).fill(BigInt(AUDIO_TOKEN)), 258883n];
        expect(input_ids.tolist()).toEqual([
          [2n, ...block(325), 1n],
          [2n, ...block(50), 1n, ...Array(275).fill(0n)],
        ]);
        expect(input_features.dims).toEqual([2, 1299, 128]);
        expect(input_features.mean().item()).toBeCloseTo(-307566.9872486383 / (2 * 1299 * 128), 6);
        expect(input_features_mask.dims).toEqual([2, 1299]);
        expect(input_features_mask.tolist().map((x) => x.filter(Boolean).length)).toEqual([1299, 199]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "video",
      async () => {
        const { input_ids, pixel_values_videos, video_position_ids, num_frames_per_video, num_soft_tokens_per_video } = await processor(null, null, null, await pan(cats, 3));
        expect(input_ids.dims).toEqual([1, 371]);
        expect(count(input_ids, VIDEO_TOKEN)).toEqual([363]);
        expect(count(input_ids, 255999)).toEqual([3]); // One block per frame
        expect(num_frames_per_video).toEqual([3]);
        expect(num_soft_tokens_per_video).toEqual([121]);
        expect(pixel_values_videos.dims).toEqual([3, 1260, 768]);
        expect(pixel_values_videos.mean().item()).toBeCloseTo(1299828.5607564594 / (3 * 1260 * 768), 3);
        expect(video_position_ids.sum().item()).toEqual(103518n);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "video with more than max_frames frames",
      async () => {
        const { input_ids, pixel_values_videos, num_frames_per_video } = await processor(null, null, null, await pan(cats, 34));
        expect(num_frames_per_video).toEqual([32]);
        expect(count(input_ids, VIDEO_TOKEN)).toEqual([32 * 121]);
        expect(pixel_values_videos.dims).toEqual([32, 1260, 768]);
        expect(pixel_values_videos.mean().item()).toBeCloseTo(13814942.472051758 / (32 * 1260 * 768), 3);

        // The frames are too similar for the mean to tell which were kept: `np.linspace(0, 33, 32).astype(int)`
        const frames = Array.from({ length: 34 }, (_, i) => i);
        expect(processor.video_processor.sample_frames(frames)).toEqual([...frames.slice(0, 16), ...frames.slice(17, 32), 33]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "text with every modality",
      async () => {
        const text = ["<|image|> A photo.", "Listen to <|audio|>, watch <|video|> and look at <|image|>."];
        const { input_ids, attention_mask, pixel_values, pixel_values_videos, input_features } = await processor(text, [[cats], [square]], [mlk_2s], [await pan(cats, 3)]);
        expect(input_ids.dims).toEqual([2, 692]);
        expect(attention_mask.sum(-1).tolist()).toEqual([273n, 692n]);
        expect(input_ids.slice(null, [0, 4]).tolist()).toEqual([
          [2n, 255999n, 258880n, 258880n],
          [2n, 27752n, 531n, 236743n],
        ]);
        expect(count(input_ids, IMAGE_TOKEN)).toEqual([266, 256]);
        expect(count(input_ids, AUDIO_TOKEN)).toEqual([0, 50]);
        expect(count(input_ids, VIDEO_TOKEN)).toEqual([0, 363]);
        expect(pixel_values.dims).toEqual([2, 2520, 768]);
        expect(pixel_values_videos.dims).toEqual([3, 1260, 768]);
        expect(input_features.dims).toEqual([1, 199, 128]);
      },
      MAX_TEST_EXECUTION_TIME,
    );

    it(
      "mismatched inputs",
      async () => {
        await expect(processor(["<|image|> A photo."])).rejects.toThrow("Found 1 <|image|> tokens in the text, but 0 inputs were passed.");
        await expect(processor(["A photo."], cats)).rejects.toThrow("Found 0 <|image|> tokens in the text, but 1 inputs were passed.");
        await expect(processor(null, [[cats], [cats]], [mlk_2s])).rejects.toThrow("Received a different number of samples per modality.");
      },
      MAX_TEST_EXECUTION_TIME,
    );
  });
};
