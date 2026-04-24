import http from "node:http";
import fs from "node:fs/promises";
import path from "node:path";
import { chromium } from "playwright";

const ROOT = path.resolve(process.cwd());
const MODEL_ID = process.env.SMOLVLM_MODEL_ID ?? "HuggingFaceTB/SmolVLM2-500M-Video-Instruct";
const MAX_NEW_TOKENS = Number(process.env.SMOLVLM_MAX_NEW_TOKENS ?? "64");
const NUM_FRAMES = Number(process.env.SMOLVLM_NUM_FRAMES ?? "8");
const VIDEO_URL =
  process.env.SMOLVLM_VIDEO_URL ??
  "https://huggingface.co/datasets/nielsr/video-demo/resolve/main/eating_spaghetti.mp4";

const MIMES = {
  ".js": "application/javascript",
  ".mjs": "application/javascript",
  ".html": "text/html",
  ".json": "application/json",
  ".wasm": "application/wasm",
  ".map": "application/json",
};

const PAGE_HTML = `<!DOCTYPE html>
<html>
<body>
<pre id="log"></pre>
<script type="module">
  const logEl = document.getElementById("log");
  const log = (...a) => { const s = a.map(String).join(" "); logEl.textContent += s + "\\n"; console.log(s); };
  window.addEventListener("error", (e) => console.error("PAGE_ERROR:", e.error?.stack || e.message));
  window.addEventListener("unhandledrejection", (e) => console.error("PAGE_REJECTION:", e.reason?.stack || String(e.reason)));

  try {
    const { AutoProcessor, AutoModelForImageTextToText, load_video, env } = await import("/dist/transformers.js");
    env.backends.onnx.wasm.proxy = false;
    const MODEL_ID = ${JSON.stringify(MODEL_ID)};
    const NUM_FRAMES = ${NUM_FRAMES};
    const MAX_NEW_TOKENS = ${MAX_NEW_TOKENS};
    const VIDEO_URL = ${JSON.stringify(VIDEO_URL)};

    log("[browser] load_video ...");
    const video = await load_video(VIDEO_URL, { num_frames: NUM_FRAMES });
    log("[browser] decoded", video.frames.length, "frames", video.width + "x" + video.height, "dur=" + video.duration.toFixed(2) + "s");

    const messages = [{
      role: "user",
      content: [
        { type: "video", video },
        { type: "text", text: "Describe what happens in this video." },
      ],
    }];

    log("[browser] loading processor + model ...");
    const processor = await AutoProcessor.from_pretrained(MODEL_ID);
    const model = await AutoModelForImageTextToText.from_pretrained(MODEL_ID, { dtype: "q4", device: "webgpu" });

    log("[browser] processing inputs ...");
    const inputs = await processor(messages);

    log("[browser] generating ...");
    const ids = await model.generate({ ...inputs, do_sample: false, max_new_tokens: MAX_NEW_TOKENS });
    const text = processor.batch_decode(ids, { skip_special_tokens: true })[0];

    log("[browser] model class:", model.constructor.name);
    log("[browser] generated:");
    log(text);
    log("[DONE]");
    await model.dispose();
  } catch (e) {
    log("[FATAL]", e?.stack ?? String(e));
  }
</script>
</body>
</html>`;

const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, "http://x");
    if (url.pathname === "/favicon.ico") { res.statusCode = 204; res.end(); return; }
    if (url.pathname === "/" || url.pathname === "/index.html") {
      res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
      res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
      res.setHeader("Content-Type", "text/html");
      res.end(PAGE_HTML);
      return;
    }
    const filePath = path.join(ROOT, url.pathname);
    if (!filePath.startsWith(ROOT)) { res.statusCode = 403; res.end("forbidden"); return; }
    const data = await fs.readFile(filePath);
    const ext = path.extname(filePath);
    res.setHeader("Content-Type", MIMES[ext] ?? "application/octet-stream");
    res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
    res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.end(data);
  } catch (e) {
    res.statusCode = 404;
    res.end(String(e?.message ?? e));
  }
});

await new Promise((r) => server.listen(0, "127.0.0.1", r));
const port = server.address().port;
console.log(`[host] serving on http://127.0.0.1:${port}`);

const browser = await chromium.launch({ channel: "chrome", headless: true });
const context = await browser.newContext();
const page = await context.newPage();

let sawDone = false;
let sawFatal = null;
page.on("console", (msg) => {
  const text = msg.text();
  if (text.startsWith("[FATAL]")) sawFatal = text;
  if (text === "[DONE]") sawDone = true;
  console.log("[page]", text);
});
page.on("pageerror", (err) => {
  sawFatal = err.stack ?? String(err);
  console.error("[pageerror]", sawFatal);
});

const timeoutMs = Number(process.env.SMOLVLM_TIMEOUT_MS ?? "300000");
await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: "domcontentloaded" });

const t0 = Date.now();
while (!sawDone && !sawFatal && Date.now() - t0 < timeoutMs) {
  await new Promise((r) => setTimeout(r, 500));
}

await browser.close();
server.close();

if (sawFatal) {
  console.error("\n[host] FAILED:", sawFatal);
  process.exit(1);
}
if (!sawDone) {
  console.error("\n[host] TIMEOUT after", timeoutMs, "ms");
  process.exit(1);
}
console.log("\n[host] OK");
