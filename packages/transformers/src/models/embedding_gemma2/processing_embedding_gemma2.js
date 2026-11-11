import { Gemma4Processor } from '../gemma4/processing_gemma4.js';
import { EmbeddingGemma2VideoProcessor } from './video_processing_embedding_gemma2.js';
import { Tensor } from '../../utils/tensor.js';

/**
 * Groups media items per sample: a single item is one sample, a nested list is kept as is, and a flat list is either a
 * single sample (`flat_is_one_sample`) or one sample per item.
 * @param {any} items
 * @param {boolean} flat_is_one_sample
 * @returns {any[][]}
 */
function nest(items, flat_is_one_sample) {
    if (!Array.isArray(items)) return [[items]];
    if (items.every((x) => Array.isArray(x))) return items;
    return flat_is_one_sample ? [items] : items.map((x) => [x]);
}

/**
 * Replaces the `token` placeholders of `text` with `replacements`, in order across the batch.
 * @param {string[]} text
 * @param {string} token
 * @param {string[]} replacements
 * @returns {string[]}
 */
function expand_placeholders(text, token, replacements) {
    let i = 0;
    text = text.map((t) => t.replaceAll(token, () => replacements[i++] ?? ''));
    if (i !== replacements.length) {
        throw new Error(`Found ${i} ${token} tokens in the text, but ${replacements.length} inputs were passed.`);
    }
    return text;
}

export class EmbeddingGemma2Processor extends Gemma4Processor {
    constructor(config, components, chat_template) {
        super(config, components, chat_template);
        this.video_token = this.tokenizer.config.video_token;
        this.video_processor = config.video_processor && new EmbeddingGemma2VideoProcessor(config.video_processor);
    }

    /**
     * Prepares any mix of text, images, audio and videos for `EmbeddingGemma2Model`.
     *
     * Each `<|image|>`, `<|audio|>` or `<|video|>` placeholder of the text is expanded to the soft tokens of the next
     * item of that modality, in order across the batch. Without text, each sample holds the placeholders of its media
     * (images, then videos, then audio). As in Python, a flat list of images is then a single sample, and a flat list
     * of audio clips or videos is one sample per item: pass one list per sample to group them differently.
     *
     * @param {string|string[]|null} text The text, with a placeholder per media item.
     * @param {import('../../utils/image.js').RawImage|import('../../utils/image.js').RawImage[]|import('../../utils/image.js').RawImage[][]} [images]
     * @param {Float32Array|Float32Array[]|Float32Array[][]} [audio] Mono audio at the feature extractor's sampling rate (16 kHz).
     * @param {import('../../utils/video.js').RawVideo|import('../../utils/video.js').RawVideo[]|import('../../utils/video.js').RawVideo[][]} [videos]
     * @param {Object} [options] Additional options for the tokenizer.
     */
    async _call(text, images = null, audio = null, videos = null, options = {}) {
        // The media items of each sample, by placeholder
        /** @type {[string, any[][]][]} */
        const media = [
            [this.image_token, images && nest(images, true)],
            [this.video_token, videos && nest(videos, false)],
            [this.audio_token, audio && nest(audio, false)],
        ];
        if (text == null) {
            const present = media.filter(([, samples]) => samples);
            const batch_size = present[0]?.[1].length;
            if (present.some(([, samples]) => samples.length !== batch_size)) {
                throw new Error('Received a different number of samples per modality.');
            }
            text = Array.from({ length: batch_size }, (_, i) =>
                present.flatMap(([token, samples]) => Array(samples[i].length).fill(token)).join(' '),
            );
        } else if (typeof text === 'string') {
            text = [text];
        }

        const [image_list, video_list, audio_list] = media.map(([, samples]) => samples?.flat());
        const image_inputs = image_list && (await this.image_processor(image_list));
        const video_inputs = video_list && (await this.video_processor(video_list));
        const audio_inputs = audio_list && (await this._process_audio(audio_list));

        const { image_token, video_token, audio_token, boi_token, eoi_token, boa_token, eoa_token } = this;
        const replacements = {
            [image_token]: image_inputs?.num_soft_tokens_per_image.map(
                (n) => boi_token + image_token.repeat(n) + eoi_token,
            ),
            // One image-like block per frame
            [video_token]: video_inputs?.num_soft_tokens_per_video.map((n, i) =>
                (boi_token + video_token.repeat(n) + eoi_token).repeat(video_inputs.num_frames_per_video[i]),
            ),
            [audio_token]: audio_inputs?.num_soft_tokens_per_audio.map(
                (n) => boa_token + audio_token.repeat(n) + eoa_token,
            ),
        };
        for (const [token, strings] of Object.entries(replacements)) {
            text = expand_placeholders(text, token, strings ?? []);
        }

        return {
            ...this.tokenizer(text, { padding: true, ...options }),
            ...image_inputs,
            ...video_inputs,
            ...audio_inputs,
        };
    }

    /**
     * Extracts the features of each clip and pads them to the longest one.
     * @param {Float32Array[]} clips
     */
    async _process_audio(clips) {
        const outputs = [];
        for (const clip of clips) {
            outputs.push(await this.feature_extractor(clip));
        }
        const num_frames = Math.max(...outputs.map((x) => x.input_features.dims[1]));
        const feature_size = outputs[0].input_features.dims[2];

        const input_features = new Float32Array(clips.length * num_frames * feature_size);
        const input_features_mask = new Uint8Array(clips.length * num_frames);
        outputs.forEach((x, i) => {
            input_features.set(x.input_features.data, i * num_frames * feature_size);
            input_features_mask.set(x.input_features_mask.data, i * num_frames);
        });

        return {
            input_features: new Tensor('float32', input_features, [clips.length, num_frames, feature_size]),
            input_features_mask: new Tensor('bool', input_features_mask, [clips.length, num_frames]),
            // The audio encoder keeps every 4th frame (two stride-2 convolutions)
            num_soft_tokens_per_audio: outputs.map(({ input_features_mask: { data } }) => {
                let num_tokens = 0;
                for (let i = 0; i < data.length; i += 4) num_tokens += Number(data[i]);
                return num_tokens;
            }),
        };
    }
}
