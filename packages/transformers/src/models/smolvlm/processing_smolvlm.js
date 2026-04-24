import { Idefics3Processor, _prompt_single_image } from '../idefics3/processing_idefics3.js';
import { load_video, RawVideo } from '../../utils/video.js';

/**
 * @typedef {import('../../image_processors_utils.js').ImageProcessorConfig & {
 *   video_sampling?: { fps?: number, max_frames?: number, video_size?: { longest_edge?: number } },
 * }} SmolVLMImageProcessorConfig
 */

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
        const vs = config.video_sampling ?? { fps: 1, max_frames: 64 };

        let v = videos;
        if (typeof v === 'string' || (typeof Blob !== 'undefined' && v instanceof Blob)) {
            v = await load_video(v, { fps: vs.fps, num_frames: vs.max_frames });
        }
        if (!(v instanceof RawVideo)) {
            throw new Error('Expected a RawVideo, URL, path, or Blob for the video input.');
        }

        const frames = v.frames.map((f) => f.image);
        const timestamps = v.frames.map((f) => f.timestamp);
        // Idefics3ImageProcessor performs the two-pass resize itself:
        //   1) preprocess() → longest-edge aspect-preserving via this.size (video_sampling.video_size if set)
        //   2) do_image_splitting:false → square to max_image_size.longest_edge
        const vision = await this.image_processor(frames, {
            ...options,
            do_image_splitting: false,
            ...(vs.video_size ? { size: vs.video_size } : {}),
        });

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
