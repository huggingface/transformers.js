import { Callable } from '../../utils/generic.js';
import { Gemma4ImageProcessor } from '../gemma4/image_processing_gemma4.js';

/**
 * Processes videos as sequences of frames, each going through the Gemma 4 image processor with the video budget of
 * soft tokens (`max_soft_tokens`). Frames are used as decoded (e.g., `load_video(url, { fps: 1 })`), and videos with
 * more than `max_frames` frames are subsampled according to `overflow_strategy` ('uniform' or 'truncate').
 */
export class EmbeddingGemma2VideoProcessor extends Callable {
    /** @param {Record<string, any>} config */
    constructor(config) {
        super();
        this.config = config;
        this.frame_processor = new Gemma4ImageProcessor(config);
        this.max_frames = config.max_frames ?? 32;
        this.overflow_strategy = config.overflow_strategy ?? 'uniform';
    }

    /**
     * @template T
     * @param {T[]} frames
     * @returns {T[]} At most `max_frames` of the frames.
     */
    sample_frames(frames) {
        const num_frames = frames.length;
        const { max_frames } = this;
        if (num_frames <= max_frames) return frames;
        if (this.overflow_strategy === 'truncate') return frames.slice(0, max_frames);

        // `np.linspace(0, num_frames - 1, max_frames).astype(int)`
        const step = max_frames > 1 ? (num_frames - 1) / (max_frames - 1) : 0;
        return Array.from(
            { length: max_frames },
            (_, i) => frames[i > 0 && i === max_frames - 1 ? num_frames - 1 : Math.floor(i * step)],
        );
    }

    /**
     * @param {import('../../utils/video.js').RawVideo|import('../../utils/video.js').RawVideo[]} videos
     */
    async _call(videos) {
        if (!Array.isArray(videos)) {
            videos = [videos];
        }
        const frames = videos.map((video) => this.sample_frames(video.frames.map((frame) => frame.image)));
        const { pixel_values, image_position_ids, num_soft_tokens_per_image } = await this.frame_processor(
            frames.flat(),
        );

        let num_previous_frames = 0;
        return {
            // The frames of all videos, concatenated
            pixel_values_videos: pixel_values,
            video_position_ids: image_position_ids,
            num_frames_per_video: frames.map((x) => x.length),
            // The frames of a video have the same size, hence the same number of soft tokens
            num_soft_tokens_per_video: frames.map(
                (x) => num_soft_tokens_per_image[(num_previous_frames += x.length) - 1],
            ),
        };
    }
}
