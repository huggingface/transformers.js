import {
  AutoProcessor,
  AutoModelForImageTextToText,
  load_video,
} from "../src/transformers.js";

const MODEL_ID =
  process.env.SMOLVLM_MODEL_ID ?? "HuggingFaceTB/SmolVLM2-500M-Video-Instruct";
const MAX_NEW_TOKENS = Number(process.env.SMOLVLM_MAX_NEW_TOKENS ?? "64");
const NUM_FRAMES = Number(process.env.SMOLVLM_NUM_FRAMES ?? "8");

// Stable sample video used in official transformers tests.
const VIDEO_URL =
  process.env.SMOLVLM_VIDEO_URL ??
  "https://huggingface.co/datasets/nielsr/video-demo/resolve/main/eating_spaghetti.mp4";

console.log(`Video: ${VIDEO_URL}`);
console.log(`Decoding ${NUM_FRAMES} frames via @napi-rs/webcodecs...`);
const video = await load_video(VIDEO_URL, { num_frames: NUM_FRAMES });
console.log(
  `Decoded ${video.frames.length} frames @ ${video.width}x${video.height}, duration ${video.duration.toFixed(2)}s`,
);

const messages = [
  {
    role: "user",
    content: [
      { type: "video", video },
      { type: "text", text: "Describe what happens in this video." },
    ],
  },
];

console.log(`Loading processor and model from: ${MODEL_ID}`);
const processor = await AutoProcessor.from_pretrained(MODEL_ID);
const model = await AutoModelForImageTextToText.from_pretrained(MODEL_ID, {
  dtype: "fp32",
});

console.log("Processing video inputs...");
const inputs = await processor(messages);

console.log("Generating response...");
const generated_ids = await model.generate({
  ...inputs,
  do_sample: false,
  max_new_tokens: MAX_NEW_TOKENS,
});

const generated_text = processor.batch_decode(generated_ids, {
  skip_special_tokens: true,
})[0];

console.log("Model class:", model.constructor.name);
console.log("Generated text:\n", generated_text);

await model.dispose();
