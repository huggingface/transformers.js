import { RawImage } from './image.js';
import { env, apis } from '../env.js';

export class RawVideoFrame {
    /**
     * @param {RawImage} image
     * @param {number} timestamp
     */
    constructor(image, timestamp) {
        this.image = image;
        this.timestamp = timestamp;
    }
}

export class RawVideo {
    /**
     * @param {RawVideoFrame[]|RawImage[]} frames
     * @param {number} duration
     */
    constructor(frames, duration) {
        if (frames.length > 0 && frames[0] instanceof RawImage) {
            // Assume uniform timestamps
            frames = frames.map((image, i) => new RawVideoFrame(image, ((i + 1) / (frames.length + 1)) * duration));
        }
        this.frames = /** @type {RawVideoFrame[]} */ (frames);
        this.duration = duration;
    }

    get width() {
        return this.frames[0].image.width;
    }
    get height() {
        return this.frames[0].image.height;
    }

    get fps() {
        return this.frames.length / this.duration;
    }
}

/**
 * Loads a video.
 *
 * In browser environments, uses HTMLVideoElement + canvas to seek and sample frames.
 * In Node.js environments, uses the optional peer dependency `@napi-rs/webcodecs`
 * which exposes the W3C WebCodecs API on top of FFmpeg (install via
 * `npm install @napi-rs/webcodecs`).
 *
 * @param {string|Blob|ArrayBuffer|Uint8Array|HTMLVideoElement} src The video to process.
 * @param {Object} [options] Optional parameters.
 * @param {number} [options.num_frames=null] Max number of frames to sample.
 * @param {number} [options.fps=null] Target frames per second to sample.
 * @param {number} [options.skip_secs=1] Seconds to skip from start/end when video is long enough.
 *
 * @returns {Promise<RawVideo>} The loaded video.
 */
export async function load_video(src, { num_frames = null, fps = null, skip_secs = 1 } = {}) {
    if (num_frames == null && fps == null) {
        throw new Error('Either num_frames or fps must be provided.');
    }

    if (apis.IS_BROWSER_ENV || apis.IS_WEBWORKER_ENV) {
        return await loadVideoBrowser(src, { num_frames, fps, skip_secs });
    }
    if (apis.IS_NODE_ENV) {
        return await loadVideoNode(src, { num_frames, fps, skip_secs });
    }
    throw new Error('`load_video` is not supported in this environment.');
}

/**
 * Port of transformers SmolVLMVideoProcessor.sample_frames to the time-domain:
 *   - desired = min(round(fps * duration), num_frames), clamped to >= 1
 *   - if skip_secs > 0 AND (duration - 2*skip_secs) > (num_frames * fps): trim both ends by skip_secs
 *   - uniform linspace(start, end, desired); dedupe times rounded to μs (≈ np.unique on indices).
 */
function computeSampleTimes(duration, { num_frames, fps, skip_secs = 1 }) {
    let desired;
    if (num_frames != null && fps != null) {
        desired = Math.min(Math.round(fps * duration), num_frames);
    } else if (fps != null) {
        desired = Math.round(fps * duration);
    } else {
        desired = num_frames;
    }
    desired = Math.max(1, desired);

    let start = 0;
    let end = duration;
    if (skip_secs > 0 && num_frames != null && fps != null
        && (duration - 2 * skip_secs) > (num_frames * fps)) {
        start = skip_secs;
        end = duration - skip_secs;
    }

    const times = [];
    const seen = new Set();
    for (let i = 0; i < desired; ++i) {
        const t = desired === 1 ? (start + end) / 2 : start + (i * (end - start)) / (desired - 1);
        const key = Math.round(t * 1e6);
        if (seen.has(key)) continue;
        seen.add(key);
        times.push(t);
    }
    return times;
}

async function loadVideoBrowser(src, { num_frames, fps, skip_secs = 1 }) {
    // TODO: Support efficiently loading all frames using the WebCodecs API.
    // Specfically, https://developer.mozilla.org/en-US/docs/Web/API/VideoDecoder
    const frames = [];

    const video = document.createElement('video');
    video.crossOrigin = 'anonymous';
    video.muted = true; // mute to allow autoplay and seeking

    if (typeof src === 'string') {
        video.src = src;
    } else if (src instanceof Blob) {
        video.src = URL.createObjectURL(src);
    } else if (src instanceof HTMLVideoElement) {
        video.src = src.src;
    } else {
        throw new Error('Invalid URL or video element provided.');
    }
    // Wait for metadata to load to obtain duration
    await new Promise((resolve) => (video.onloadedmetadata = resolve));

    if (video.seekable.start(0) === video.seekable.end(0)) {
        // Fallback: Download entire video if not seekable
        const response = await env.fetch(video.src);
        const blob = await response.blob();
        video.src = URL.createObjectURL(blob);
        await new Promise((resolve) => (video.onloadedmetadata = resolve));
    }

    const duration = video.duration;
    const sampleTimes = computeSampleTimes(duration, { num_frames, fps, skip_secs });

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    for (const t of sampleTimes) {
        video.currentTime = t;
        await new Promise((resolve) => {
            video.onseeked = resolve;
        });
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const frameData = new RawImage(imageData.data, canvas.width, canvas.height, 4);

        const frame = new RawVideoFrame(frameData, t);
        frames.push(frame);
    }

    // Clean up video element.
    video.remove();

    return new RawVideo(frames, duration);
}

async function readSrcToBytes(src) {
    if (typeof src === 'string') {
        if (/^https?:/i.test(src)) return new Uint8Array(await (await fetch(src)).arrayBuffer());
        return new Uint8Array(await (await import('node:fs/promises')).readFile(src));
    }
    if (src instanceof ArrayBuffer) return new Uint8Array(src);
    if (ArrayBuffer.isView(src)) return new Uint8Array(src.buffer, src.byteOffset, src.byteLength);
    return new Uint8Array(await src.arrayBuffer()); // Blob-like
}

async function loadVideoNode(src, { num_frames, fps, skip_secs = 1 }) {
    let wc;
    try { wc = await import('@napi-rs/webcodecs'); }
    catch { throw new Error('Node video decoding requires `npm install @napi-rs/webcodecs`.'); }

    const bytes = await readSrcToBytes(src);
    // Container sniff: EBML → MKV/WebM, else assume MP4 (has 'ftyp' @ offset 4).
    const isMkv = bytes[0] === 0x1a && bytes[1] === 0x45 && bytes[2] === 0xdf && bytes[3] === 0xa3;
    const Demuxer = isMkv ? (wc.MkvDemuxer ?? wc.WebMDemuxer) : wc.Mp4Demuxer;

    let targetsUs = [];
    /** @type {Array<{f: any, d: number} | null>} */
    let best = [];
    let err = null;

    const decoder = new wc.VideoDecoder({
        // Sync: keep or close up front; defer async pixel copy until after flush.
        output: (f) => {
            const ts = Number(f.timestamp);
            let i = 0, d = Infinity;
            for (let j = 0; j < targetsUs.length; j++) {
                const dj = Math.abs(ts - targetsUs[j]);
                if (dj < d) { d = dj; i = j; }
            }
            if (!best[i] || d < best[i].d) { best[i]?.f.close?.(); best[i] = { f, d }; }
            else f.close?.();
        },
        error: (e) => (err = e),
    });

    const demuxer = new Demuxer({
        videoOutput: (c) => decoder.decode(c),
        error: (e) => (err = e),
    });
    await demuxer.loadBuffer(bytes);

    const duration = Number(demuxer.duration) / 1e6;
    const sampleTimes = computeSampleTimes(duration, { num_frames, fps, skip_secs });
    targetsUs = sampleTimes.map(t => Math.round(t * 1e6));
    best = Array(sampleTimes.length).fill(null);

    decoder.configure(demuxer.videoDecoderConfig);
    demuxer.demux();
    await decoder.flush();
    demuxer.close();
    decoder.close();
    if (err) throw err;

    // Backfill any empty buckets from nearest populated neighbour.
    for (let i = 0; i < best.length; i++) if (!best[i]) {
        let near = -1;
        for (let j = 0; j < best.length; j++)
            if (best[j] && (near < 0 || Math.abs(i - j) < Math.abs(i - near))) near = j;
        if (near < 0) throw new Error('No frames decoded.');
        best[i] = best[near];
    }

    // Copy pixels (once per unique frame), then close.
    const imgs = new Map();
    for (const { f } of /** @type {{f: any}[]} */ (best)) {
        if (imgs.has(f)) continue;
        const buf = new Uint8Array(f.allocationSize({ format: 'RGBA' }));
        await f.copyTo(buf, { format: 'RGBA' });
        const { codedWidth: w, codedHeight: h } = f;
        imgs.set(f, new RawImage(new Uint8ClampedArray(buf.buffer, buf.byteOffset, w * h * 4), w, h, 4));
    }
    for (const f of imgs.keys()) f.close?.();

    const frames = best.map(({ f }, i) => new RawVideoFrame(imgs.get(f), sampleTimes[i]));
    return new RawVideo(frames, duration);
}
