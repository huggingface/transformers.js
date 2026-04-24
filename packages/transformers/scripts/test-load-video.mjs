import { load_video } from "../src/transformers.js";

const VIDEO_URL =
  process.env.VIDEO_URL ??
  "https://huggingface.co/datasets/nielsr/video-demo/resolve/main/eating_spaghetti.mp4";
const NUM_FRAMES = Number(process.env.NUM_FRAMES ?? "8");

console.log(`Fetching & decoding ${NUM_FRAMES} frames from ${VIDEO_URL}...`);
const t0 = Date.now();
const video = await load_video(VIDEO_URL, { num_frames: NUM_FRAMES });
const dt = Date.now() - t0;

console.log(
  `OK: ${video.frames.length} frames, ${video.width}x${video.height}, duration=${video.duration.toFixed(2)}s, elapsed=${dt}ms`,
);
for (const f of video.frames) {
  console.log(
    `  t=${f.timestamp.toFixed(3)}s  ${f.image.width}x${f.image.height}x${f.image.channels}  bytes=${f.image.data.length}`,
  );
}
