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
 * @param {number} [options.num_frames=null] The number of frames to sample uniformly.
 * @param {number} [options.fps=null] The number of frames to sample per second.
 *
 * @returns {Promise<RawVideo>} The loaded video.
 */
export async function load_video(src, { num_frames = null, fps = null } = {}) {
    if (num_frames == null && fps == null) {
        throw new Error('Either num_frames or fps must be provided.');
    }

    if (apis.IS_BROWSER_ENV || apis.IS_WEBWORKER_ENV) {
        return await loadVideoBrowser(src, { num_frames, fps });
    }
    if (apis.IS_NODE_ENV) {
        return await loadVideoNode(src, { num_frames, fps });
    }
    throw new Error('`load_video` is not supported in this environment.');
}

function computeSampleTimes(duration, { num_frames, fps }) {
    let count, step;
    if (num_frames != null) {
        count = num_frames;
        step = num_frames === 1 ? 0 : duration / (num_frames - 1);
    } else {
        step = 1 / fps;
        count = Math.floor(duration / step);
    }
    const times = [];
    for (let i = 0; i < count; ++i) {
        times.push(num_frames === 1 ? duration / 2 : i * step);
    }
    return times;
}

async function loadVideoBrowser(src, { num_frames, fps }) {
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
    const sampleTimes = computeSampleTimes(duration, { num_frames, fps });

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

async function loadVideoNode(src, { num_frames, fps }) {
    let wc;
    try {
        wc = await import('@napi-rs/webcodecs');
    } catch (_e) {
        throw new Error(
            'Node.js video decoding requires the optional peer dependency `@napi-rs/webcodecs`. ' +
            'Install it with:\n\n' +
            '    npm install @napi-rs/webcodecs\n\n' +
            'Alternatively, pre-decode your frames and pass a `RawVideo` instance directly to the processor.',
        );
    }
    const { Mp4Demuxer, WebMDemuxer, MkvDemuxer, VideoDecoder } = wc;

    const bytes = await readSrcToBytes(src);
    // Magic byte sniff: MP4 has 'ftyp' at offset 4; Matroska/WebM starts with EBML 0x1A45DFA3.
    const isMkv = bytes[0] === 0x1a && bytes[1] === 0x45 && bytes[2] === 0xdf && bytes[3] === 0xa3;
    const Demuxer = isMkv ? (MkvDemuxer ?? WebMDemuxer) : Mp4Demuxer;

    /** @type {number[]} */
    let targetsUs = [];
    /** @type {Array<{frame: any, dist: number, ts: number} | null>} */
    let best = [];
    let framesSeen = 0;
    let decoderError = null;

    const decoder = new VideoDecoder({
        // Synchronous: decide keep-or-close up front; defer pixel copy until after flush.
        output: (frame) => {
            framesSeen++;
            const ts = Number(frame.timestamp);
            let bestIdx = 0;
            let bestDist = Infinity;
            for (let i = 0; i < targetsUs.length; i++) {
                const d = Math.abs(ts - targetsUs[i]);
                if (d < bestDist) {
                    bestDist = d;
                    bestIdx = i;
                }
            }
            const slot = best[bestIdx];
            if (!slot || bestDist < slot.dist) {
                slot?.frame.close?.();
                best[bestIdx] = { frame, dist: bestDist, ts };
            } else {
                frame.close?.();
            }
        },
        error: (e) => {
            decoderError = e;
        },
    });

    const demuxer = new Demuxer({
        videoOutput: (chunk) => decoder.decode(chunk),
        error: (e) => {
            decoderError = e;
        },
    });

    await demuxer.loadBuffer(bytes);

    const duration = Number(demuxer.duration) / 1e6; // μs → s
    if (!Number.isFinite(duration) || duration <= 0) {
        throw new Error(`Invalid duration reported by demuxer: ${demuxer.duration}`);
    }

    const sampleTimes = computeSampleTimes(duration, { num_frames, fps });
    targetsUs = sampleTimes.map((t) => Math.round(t * 1e6));
    best = sampleTimes.map(() => null);

    decoder.configure(demuxer.videoDecoderConfig);
    demuxer.demux();
    await decoder.flush();
    demuxer.close();
    decoder.close();

    if (decoderError) throw decoderError;
    if (framesSeen === 0) {
        throw new Error('No frames were decoded. The container/codec may be unsupported by @napi-rs/webcodecs.');
    }

    // Fill any empty buckets with the spatially-nearest populated one (can happen when
    // a sample time falls outside the decoded-frame timestamp range).
    for (let i = 0; i < best.length; i++) {
        if (best[i]) continue;
        let nearest = -1;
        let dist = Infinity;
        for (let j = 0; j < best.length; j++) {
            if (best[j] && Math.abs(i - j) < dist) {
                dist = Math.abs(i - j);
                nearest = j;
            }
        }
        if (nearest < 0) throw new Error('No decoded frames available to fill sample buckets.');
        best[i] = best[nearest];
    }

    // Now copy pixel data for the surviving frames.
    const frames = await Promise.all(best.map(async (slot, i) => {
        const { frame, ts } = /** @type {{frame: any, ts: number}} */ (slot);
        const size = frame.allocationSize({ format: 'RGBA' });
        const buf = new Uint8Array(size);
        await frame.copyTo(buf, { format: 'RGBA' });
        const w = frame.codedWidth;
        const h = frame.codedHeight;
        const pixels = new Uint8ClampedArray(buf.buffer, buf.byteOffset, w * h * 4);
        const image = new RawImage(new Uint8ClampedArray(pixels), w, h, 4);
        return new RawVideoFrame(image, sampleTimes[i]);
    }));

    // Close any kept frames (deduplicated by identity).
    const closed = new Set();
    for (const slot of best) {
        if (slot && !closed.has(slot.frame)) {
            slot.frame.close?.();
            closed.add(slot.frame);
        }
    }

    return new RawVideo(frames, duration);
}
