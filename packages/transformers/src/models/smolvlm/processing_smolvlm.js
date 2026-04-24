import { Idefics3Processor, _prompt_single_image } from '../idefics3/processing_idefics3.js';
import { load_video, RawVideo } from '../../utils/video.js';

/**
 * @typedef {import('../../image_processors_utils.js').ImageProcessorConfig & {
 *   video_sampling?: { fps?: number, max_frames?: number },
 *   size?: { longest_edge?: number },
 *   max_image_size?: { longest_edge?: number },
 * }} SmolVLMImageProcessorConfig
 */

const MAX_IMAGE_SIZE = 4096; // Absolute upper bound for resize target.

/**
 * Compute an output size with the longest edge equal to `maxSide`,
 * preserving aspect ratio and forcing even dimensions.
 */
function aspectPreservingSize(width, height, maxSide) {
    maxSide = Math.min(MAX_IMAGE_SIZE, maxSide);
    const aspect = width / height;
    let w, h;
    if (width >= height) {
        w = maxSide;
        h = Math.floor(w / aspect);
        if (h % 2) h += 1;
    } else {
        h = maxSide;
        w = Math.floor(h * aspect);
        if (w % 2) w += 1;
    }
    return { width: Math.max(1, w), height: Math.max(1, h) };
}

const DEFAULT_VIDEO_INTRO =
    'You are provided the following series of {frame_count} frames from a {video_duration} [H:MM:SS] video.\n';
const FRAME_TIMESTAMP_MESSAGE = '\nFrame from {timestamp}:';
const DEFAULT_MEDIA_OUTTRO = '\n\n';

// Chat template used when the conversation contains a <video> block.
const DEFAULT_CHAT_TEMPLATE =
    "<|im_start|>{% for message in messages %}{{message['role'] | capitalize}}" +
    "{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}" +
    "{% for line in message['content'] %}" +
    "{% if line['type'] == 'text' %}{{line['text']}}" +
    "{% elif line['type'] == 'image' %}{{ '<image>' }}" +
    "{% elif line['type'] == 'video' %}{{ '<video>' }}{% endif %}{% endfor %}" +
    "<end_of_utterance>\n{% endfor %}" +
    "{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}";

const NUM_WORDS = [
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen',
];
const TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'];

function num2words(n) {
    if (n < 20) return NUM_WORDS[n];
    if (n < 100) return TENS[Math.floor(n / 10)] + (n % 10 ? '-' + NUM_WORDS[n % 10] : '');
    return String(n);
}

function formatDuration(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    const mm = String(m).padStart(2, '0');
    const ss = String(s).padStart(2, '0');
    return h > 0 ? `${h}:${mm}:${ss}` : `${m}:${ss}`;
}

function pickMedia(block, kind) {
    return block.type === kind ? (block[kind] ?? block.url ?? block.path ?? null) : null;
}

export class SmolVLMProcessor extends Idefics3Processor {
    video_token = '<video>';

    apply_chat_template(conversation, options = {}) {
        const convs = Array.isArray(conversation) && conversation[0]?.role ? [conversation] : conversation;
        const hasVideo = convs.some((c) =>
            c.some((m) => Array.isArray(m.content) && m.content.some((b) => b.type === 'video')),
        );
        if (hasVideo && !options.chat_template && !this.chat_template) {
            options.chat_template = DEFAULT_CHAT_TEMPLATE;
        }
        return super.apply_chat_template(conversation, options);
    }

    /**
     * @param {string|string[]|Array<{role:string,content:any}>} text
     * @param {any} images
     * @param {Object} options
     */
    async _call(text = null, images = null, options = {}) {
        // Conversation form: extract images/videos from content blocks, then template.
        if (Array.isArray(text) && text[0] && typeof text[0] === 'object' && 'role' in text[0]) {
            const msgs = /** @type {Array<{role: string, content: any}>} */ (text);
            const imgs = [];
            const vids = [];
            for (const m of msgs) {
                if (!Array.isArray(m.content)) continue;
                for (const b of m.content) {
                    const i = pickMedia(b, 'image');
                    const v = pickMedia(b, 'video');
                    if (i != null) imgs.push(i);
                    if (v != null) vids.push(v);
                }
            }
            images ??= imgs.length ? imgs : null;
            options.videos ??= vids[0] ?? null;

            const stripped = msgs.map((m) => ({
                ...m,
                content: Array.isArray(m.content)
                    ? m.content.map(({ image, video, ...rest }) => rest)
                    : m.content,
            }));
            text = /** @type {string} */ (this.apply_chat_template(stripped, { add_generation_prompt: true }));
        }

        const videos = options.videos ?? null;
        if (videos != null) {
            return await this._processVideo(/** @type {string|string[]} */ (text), videos, options);
        }

        // Image-only or text-only: delegate to Idefics3Processor.
        return await super._call(/** @type {string|string[]} */ (text), images, options);
    }

    async _processVideo(text, videos, options) {
        const config = /** @type {SmolVLMImageProcessorConfig} */ (this.image_processor.config);

        let v = videos;
        if (typeof v === 'string' || (typeof Blob !== 'undefined' && v instanceof Blob)) {
            const cfg = config.video_sampling ?? { fps: 1, max_frames: 64 };
            v = await load_video(v, { fps: cfg.fps, num_frames: cfg.max_frames });
        }
        if (!(v instanceof RawVideo)) {
            throw new Error('Expected a RawVideo, URL, path, or Blob for the video input.');
        }

        // Two-pass resize:
        //   1) longest-edge → size.longest_edge, aspect-preserving, even dims (here, on the RawImage)
        //   2) square → max_image_size.longest_edge (by Idefics3ImageProcessor with do_image_splitting:false)
        const longestEdge = config.size?.longest_edge;
        const rawFrames = v.frames.map((f) => f.image);
        const frames = longestEdge
            ? await Promise.all(rawFrames.map(async (img) => {
                const { width, height } = aspectPreservingSize(img.width, img.height, longestEdge);
                return (img.width === width && img.height === height) ? img : await img.resize(width, height);
            }))
            : rawFrames;
        const timestamps = v.frames.map((f) => f.timestamp);
        const vision = await this.image_processor(frames, { ...options, do_image_splitting: false });

        const texts = (Array.isArray(text) ? text : [text]).map((s) =>
            this._expandVideoTokens(s, frames.length, v.duration, timestamps),
        );
        return { ...this.tokenizer(texts), ...vision };
    }

    _expandVideoTokens(text, frameCount, duration, timestamps) {
        const framePlaceholder = _prompt_single_image(
            this.config.image_seq_len ?? 169,
            this.fake_image_token,
            this.image_token,
            this.global_img_token,
        );
        while (text.includes(this.video_token)) {
            let prompt = DEFAULT_VIDEO_INTRO
                .replace('{frame_count}', num2words(frameCount))
                .replace('{video_duration}', formatDuration(duration));
            for (const t of timestamps) {
                const mm = String(Math.floor(t / 60)).padStart(2, '0');
                const ss = String(Math.floor(t % 60)).padStart(2, '0');
                prompt += FRAME_TIMESTAMP_MESSAGE.replace('{timestamp}', `${mm}:${ss}`) + framePlaceholder;
            }
            prompt += DEFAULT_MEDIA_OUTTRO;
            text = text.replace(this.video_token, prompt);
        }
        return text;
    }
}
