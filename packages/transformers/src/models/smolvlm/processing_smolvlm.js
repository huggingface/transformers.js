import { Processor } from '../../processing_utils.js';
import { AutoImageProcessor } from '../auto/image_processing_auto.js';
import { AutoTokenizer } from '../auto/tokenization_auto.js';
import { load_video, RawVideo } from '../../utils/video.js';

/**
 * Helper function to convert numbers to words (limited to common frame counts).
 * @param {number} n
 * @returns {string}
 */
function num2words(n) {
    const units = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen'];
    const tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'];
    if (n < 20) return units[n];
    if (n < 100) return tens[Math.floor(n / 10)] + (n % 10 !== 0 ? '-' + units[n % 10] : '');
    return n.toString(); // Fallback for large numbers
}

/**
 * Format duration in seconds to H:MM:SS
 * @param {number} seconds
 * @returns {string}
 */
function format_duration(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) {
        return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    }
    return `${m}:${s.toString().padStart(2, '0')}`;
}

/**
 * Prompt with expanded image tokens for when the image is split into patches.
 * @private
 */
function _prompt_split_image(
    image_seq_len,
    image_rows,
    image_cols,
    fake_token_around_image,
    image_token,
    global_img_token,
) {
    let text_split_images = '';
    for (let n_h = 0; n_h < image_rows; ++n_h) {
        for (let n_w = 0; n_w < image_cols; ++n_w) {
            text_split_images +=
                fake_token_around_image + `<row_${n_h + 1}_col_${n_w + 1}>` + image_token.repeat(image_seq_len);
        }
        text_split_images += '\n';
    }

    text_split_images +=
        `\n${fake_token_around_image}` +
        `${global_img_token}` +
        image_token.repeat(image_seq_len) +
        `${fake_token_around_image}`;
    return text_split_images;
}

/**
 * Prompt with expanded image tokens for a single image.
 * @private
 */
function _prompt_single_image(image_seq_len, fake_token_around_image, image_token, global_img_token) {
    return (
        `${fake_token_around_image}` +
        `${global_img_token}` +
        image_token.repeat(image_seq_len) +
        `${fake_token_around_image}`
    );
}

function get_image_prompt_string(
    image_rows,
    image_cols,
    image_seq_len,
    fake_token_around_image,
    image_token,
    global_img_token,
) {
    if (image_rows === 0 && image_cols === 0) {
        return _prompt_single_image(image_seq_len, fake_token_around_image, image_token, global_img_token);
    }
    return _prompt_split_image(
        image_seq_len,
        image_rows,
        image_cols,
        fake_token_around_image,
        image_token,
        global_img_token,
    );
}

const DEFAULT_VIDEO_INTRO = "You are provided the following series of {frame_count} frames from a {video_duration} [H:MM:SS] video.\n";
const FRAME_TIMESTAMP_MESSAGE = "\nFrame from {timestamp}:";
const DEFAULT_MEDIA_OUTTRO = "\n\n";

// The correct chat template to be used for videos
const DEFAULT_CHAT_TEMPLATE = "<|im_start|>{% for message in messages %}{{message['role'] | capitalize}}{% if message['content'][0]['type'] == 'image' %}{{':'}}{% else %}{{': '}}{% endif %}{% for line in message['content'] %}{% if line['type'] == 'text' %}{{line['text']}}{% elif line['type'] == 'image' %}{{ '<image>' }}{% elif line['type'] == 'video' %}{{ '<video>' }}{% endif %}{% endfor %}<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}";

export class SmolVLMProcessor extends Processor {
    static image_processor_class = AutoImageProcessor;
    static tokenizer_class = AutoTokenizer;
    static uses_processor_config = true;

    constructor(config, components, chat_template) {
        super(config, components, chat_template);

        this.fake_image_token = getattr(this.tokenizer, "fake_image_token", "<fake_token_around_image>");
        this.image_token = getattr(this.tokenizer, "image_token", "<image>");
        this.global_img_token = getattr(this.tokenizer, "global_image_token", "<global-img>");
        this.video_token = getattr(this.tokenizer, "video_token", "<video>");
        this.image_seq_len = this.config.image_seq_len ?? 169;
    }

    /**
     * @param {string[]} text
     * @param {number[][]} image_rows
     * @param {number[][]} image_cols
     * @returns {string[]}
     */
    expand_text_with_image_tokens(text, image_rows, image_cols) {
        const prompt_strings = [];
        for (let i = 0; i < text.length; ++i) {
            let sample = text[i];
            const sample_rows = image_rows[i];
            const sample_cols = image_cols[i];

            const image_prompt_strings = [];
            for (let j = 0; j < sample_rows.length; ++j) {
                const image_prompt_string = get_image_prompt_string(
                    sample_rows[j],
                    sample_cols[j],
                    this.image_seq_len,
                    this.fake_image_token,
                    this.image_token,
                    this.global_img_token,
                );
                image_prompt_strings.push(image_prompt_string);
            }

            const split_sample = sample.split(this.image_token);
            if (split_sample.length === 0) {
                throw new Error("The image token should be present in the text.");
            }

            let new_sample = split_sample[0];
            for (let j = 0; j < image_prompt_strings.length; ++j) {
                new_sample += image_prompt_strings[j] + (split_sample[j + 1] ?? '');
            }
            prompt_strings.push(new_sample);
        }
        return prompt_strings;
    }

    /**
     * @param {string[]} text
     * @param {any} video_inputs
     * @returns {string[]}
     */
    expand_text_with_video_tokens(text, video_inputs) {
        const { frames, duration, timestamps } = video_inputs;
        // In python, video_metadata is an iterator over batches.
        // For simplicity in JS, we assume one video per expansion batch since batching videos isn't fully supported in JS yet.
        
        const num_frames = frames.length;
        const frame_count_word = num2words(num_frames);
        const duration_str = format_duration(duration);

        const prompt_strings = [];
        for (let i = 0; i < text.length; ++i) {
            let sample = text[i];
            
            while (sample.includes(this.video_token)) {
                let video_prompt = DEFAULT_VIDEO_INTRO
                    .replace('{frame_count}', frame_count_word)
                    .replace('{video_duration}', duration_str);

                for (let j = 0; j < timestamps.length; ++j) {
                    const timestamp = timestamps[j];
                    const m = Math.floor(timestamp / 60);
                    const s = Math.floor(timestamp % 60);
                    const timestamp_str = `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;

                    const image_prompt = _prompt_single_image(
                        this.image_seq_len,
                        this.fake_image_token,
                        this.image_token,
                        this.global_img_token,
                    );
                    video_prompt += FRAME_TIMESTAMP_MESSAGE.replace('{timestamp}', timestamp_str) + image_prompt;
                }
                video_prompt += DEFAULT_MEDIA_OUTTRO;

                sample = sample.replace(this.video_token, video_prompt);
            }
            prompt_strings.push(sample);
        }
        return prompt_strings;
    }

    /**
     * @param {any} conversation
     * @param {Object} options
     */
    apply_chat_template(conversation, options = {}) {
        let conversations = conversation;
        if (!Array.isArray(conversation) || (conversation.length > 0 && conversation[0].role)) {
            conversations = [conversation];
        }

        const has_video = conversations.some(conv => 
            conv.some(m => Array.isArray(m.content) && m.content.some(c => c.type === 'video'))
        );

        if (has_video && !options.chat_template && !this.chat_template) {
            options.chat_template = DEFAULT_CHAT_TEMPLATE;
        }

        // To match python perfectly:
        if (options.processor_kwargs) {
            // Not supported exactly in JS yet
        }

        return super.apply_chat_template(conversation, options);
    }

    /**
     *
     * @param {string|string[]} text
     * @param {any} images
     * @param {Object} options
     * @returns {Promise<any>}
     */
    async _call(text = null, images = null, options = {}) {
        let videos = options.videos ?? null;

        // JS-specific behavior for when text is an array of messages
        let is_conversation = text && Array.isArray(text) && text.length > 0 && typeof text[0] === 'object' && 'role' in text[0];
        
        if (is_conversation) {
            let extracted_images = [];
            let extracted_videos = [];
            for (const message of text) {
                if (Array.isArray(message.content)) {
                    for (const block of message.content) {
                        if (block.type === 'image') {
                            if (block.image) extracted_images.push(block.image);
                            else if (block.url) extracted_images.push(block.url);
                            else if (block.path) extracted_images.push(block.path);
                        } else if (block.type === 'video') {
                            if (block.video) extracted_videos.push(block.video);
                            else if (block.url) extracted_videos.push(block.url);
                            else if (block.path) extracted_videos.push(block.path);
                        }
                    }
                }
            }
            
            if (extracted_videos.length > 0 && !videos) {
                videos = extracted_videos[0];
            }
            if (extracted_images.length > 0 && !images) {
                images = extracted_images;
            }

            // Create a copy of the conversation without the binary data to prevent the template engine from OOMing
            const text_for_template = text.map(message => ({
                ...message,
                content: Array.isArray(message.content) 
                    ? message.content.map(block => {
                        const { image, video, ...rest } = block;
                        return rest;
                    })
                    : message.content
            }));

            text = this.apply_chat_template(text_for_template, { add_generation_prompt: true });
        }

        if (text === null && images === null && videos === null) {
            throw new Error("You must provide one of `text`, `images` or `videos'.");
        }

        if (text === null && ((images === null) !== (videos !== null))) {
            throw new Error("You must specify exactly one of `images` or `videos`");
        }

        if (text !== null) {
            if (!Array.isArray(text)) {
                text = [text];
            } else if (text.length > 0 && typeof text[0] !== 'string') {
                throw new Error("Invalid input text. Please provide a string, or a list of strings");
            }

            const n_images_in_text = text.reduce((acc, sample) => acc + (sample.split(this.image_token).length - 1), 0);
            if (n_images_in_text > 0 && (images === null && videos === null)) {
                throw new Error(`We detected ${n_images_in_text} tokens in the text but no images/videos were passed`);
            }
        }

        let inputs = {};

        // Images and videos are mutually exclusive, so process one which is present
        if (images !== null) {
            const vision_inputs = await this.image_processor(images, { ...options, return_row_col_info: true });
            
            let image_rows = vision_inputs.rows;
            let image_cols = vision_inputs.cols;
            delete vision_inputs.rows;
            delete vision_inputs.cols;

            Object.assign(inputs, vision_inputs);

            if (text !== null) {
                const n_images_in_text = text.map(sample => sample.split(this.image_token).length - 1);
                
                // Assuming flat images array mapping 1:1, we don't have nested array checks in JS usually
                // but we'll try to emulate the exact rows/cols logic
                if (image_rows === undefined) {
                    image_rows = n_images_in_text.map(n_images => new Array(n_images).fill(0));
                } else if (image_rows.length !== text.length && text.length === 1) {
                    // Wrap if missing batch dimension
                    image_rows = [image_rows];
                }
                if (image_cols === undefined) {
                    image_cols = n_images_in_text.map(n_images => new Array(n_images).fill(0));
                } else if (image_cols.length !== text.length && text.length === 1) {
                    image_cols = [image_cols];
                }

                // Check lengths
                const n_images_in_images = image_rows.map(r => r.length);
                for (let i = 0; i < n_images_in_text.length; i++) {
                    if (n_images_in_images[i] !== n_images_in_text[i]) {
                        throw new Error(`The number of images in the text ${n_images_in_text[i]} and images ${n_images_in_images[i]} should be the same.`);
                    }
                }

                text = this.expand_text_with_image_tokens(text, image_rows, image_cols);
            }
        } else if (videos !== null) {
            let vision_inputs;
            if (typeof videos === 'string' || videos instanceof Blob) {
                const video_sampling = this.image_processor.config.video_sampling ?? { fps: 1, max_frames: 64 };
                videos = await load_video(videos, {
                    fps: video_sampling.fps,
                    num_frames: video_sampling.max_frames,
                });
            } else if (videos instanceof RawVideo) {
                // do nothing
            }

            if (videos instanceof RawVideo) {
                const frames = videos.frames.map(f => f.image);
                vision_inputs = await this.image_processor(frames, { ...options, do_image_splitting: false });
                vision_inputs.frames = frames;
                vision_inputs.duration = videos.duration;
                vision_inputs.timestamps = videos.frames.map(f => f.timestamp);
                // Also add video_metadata to emulate python
                vision_inputs.video_metadata = [{
                    fps: 24, // fallback
                    duration: videos.duration,
                    timestamps: videos.frames.map(f => f.timestamp),
                }];
            } else {
                // assume videos is already processed or mocked
                vision_inputs = videos;
            }

            if (text !== null) {
                const n_videos_in_text = text.map(sample => sample.split(this.video_token).length - 1);
                const n_videos_in_videos = [1]; // Assuming 1 video per sample for JS
                if (n_videos_in_videos[0] !== n_videos_in_text[0]) {
                    throw new Error(`The number of videos in the text ${n_videos_in_text} and videos ${n_videos_in_videos} should be the same.`);
                }
                text = this.expand_text_with_video_tokens(text, vision_inputs);
            }

            if (!options.return_metadata) {
                delete vision_inputs.video_metadata;
            }
            
            // For matching Python exactly: we don't output 'frames', 'duration', 'timestamps'
            delete vision_inputs.frames;
            delete vision_inputs.duration;
            delete vision_inputs.timestamps;

            Object.assign(inputs, vision_inputs);
        }

        if (text !== null) {
            const text_inputs = this.tokenizer(text, options);
            Object.assign(inputs, text_inputs);
        }

        return inputs;
    }
}

function getattr(obj, prop, def) {
    return obj[prop] ?? def;
}
