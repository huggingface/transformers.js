import { RawImage } from "./image.js";

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
            frames = frames.map((image, i) => new RawVideoFrame(image, (i + 1) / (frames.length + 1) * duration));
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
 * @param {string|Blob|HTMLVideoElement} url The video to process.
 * @param {Object} [options] Optional parameters.
 * @param {number} [options.num_frames=null] The number of frames to sample uniformly.
 * If provided, the video is seeked to the desired positions rather than processing every frame.
 * @param {number} [options.fps=null] The number of frames to sample per second.
 * If provided (and num_frames is null), the video is seeked at fixed time intervals.
 *
 * @returns {Promise<RawVideo>} The video
 */
export async function load_video(url, { num_frames = null, fps = null } = {}) {
    const frames = [];

    const video = document.createElement('video');
    if (typeof url === 'string') {
        video.src = url;
    } else if (url instanceof Blob) {
        video.src = URL.createObjectURL(url);
    } else if (url instanceof HTMLVideoElement) {
        video.src = url.src;
    } else {
        throw new Error("Invalid URL or video element provided.");
    }
    video.crossOrigin = "anonymous";
    video.muted = true; // mute to allow autoplay

    // Wait for metadata to load to obtain duration
    await new Promise((resolve) => {
        video.onloadedmetadata = resolve;
    });

    if (video.seekable.start(0) === video.seekable.end(0)) {
        // Fallback: Download entire video if not seekable
        const response = await fetch(video.src);
        const blob = await response.blob();
        video.src = URL.createObjectURL(blob);

        await new Promise((resolve) => {
            video.onloadedmetadata = resolve;
        });
    }

    const duration = video.duration;

    // Build an array of sample times based on num_frames or fps
    let sampleTimes = [];
    if (num_frames == null && fps == null) {
        throw new Error("Either num_frames or fps must be provided.");
    }

    let count, step;
    if (num_frames != null) {
        count = num_frames;
        step = num_frames === 1 ? 0 : duration / (num_frames - 1);
    } else {
        step = 1 / fps;
        count = Math.floor(duration / step);
    }

    for (let i = 0; i < count; i++) {
        sampleTimes.push(num_frames === 1 ? duration / 2 : i * step);
    }

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
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
