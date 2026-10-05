import { PreTrainedModel } from '../modeling_utils.js';
import { sessionRun } from '../session.js';
import { Tensor } from '../../utils/tensor.js';

export class EmbeddingGemma2PreTrainedModel extends PreTrainedModel {}

/**
 * The EmbeddingGemma 2 model, which embeds any mix of text, images, audio and video into a single vector.
 *
 * The vision and audio encoders turn media into soft tokens, and the text model places them at their placeholder
 * tokens, runs the bidirectional text encoder, and returns the per-token `last_hidden_state` together with the
 * mean-pooled, L2-normalized `sentence_embedding`.
 *
 * The encoders are only loaded when the config has a `vision_config` / `audio_config`. For text-only use, remove them:
 * ```javascript
 * const model_id = 'onnx-community/embeddinggemma-2-ONNX';
 * const config = await AutoConfig.from_pretrained(model_id);
 * config.vision_config = config.audio_config = null;
 * const model = await AutoModel.from_pretrained(model_id, { config });
 * ```
 */
export class EmbeddingGemma2Model extends EmbeddingGemma2PreTrainedModel {
    async forward({
        input_ids,
        attention_mask,
        pixel_values = null,
        image_position_ids = null,
        pixel_values_videos = null,
        video_position_ids = null,
        input_features = null,
        input_features_mask = null,
    }) {
        // Soft tokens of each modality present (video frames go through the vision encoder like images)
        const features = {
            image_features:
                pixel_values && (await this.encode_image({ pixel_values, pixel_position_ids: image_position_ids })),
            video_features:
                pixel_values_videos &&
                (await this.encode_image({
                    pixel_values: pixel_values_videos,
                    pixel_position_ids: video_position_ids,
                })),
            audio_features: input_features && (await this.encode_audio({ input_features, input_features_mask })),
        };

        // The text model consumes the soft tokens of each modality in placeholder order, across the whole batch (none
        // for absent modalities).
        const { image_token_id, video_token_id, audio_token_id, text_config } = /** @type {any} */ (this.config);
        const token_ids = { image: image_token_id, video: video_token_id, audio: audio_token_id };
        for (const [modality, token_id] of Object.entries(token_ids)) {
            const name = `${modality}_features`;
            features[name] ??= new Tensor('float32', new Float32Array(0), [0, text_config.hidden_size]);

            let num_tokens = 0;
            for (const id of input_ids.data) {
                if (Number(id) === token_id) ++num_tokens;
            }
            if (num_tokens !== features[name].dims[0]) {
                throw new Error(
                    `The number of ${modality} tokens (${num_tokens}) and ${modality} features (${features[name].dims[0]}) do not match.`,
                );
            }
        }
        return await sessionRun(this.sessions['model'], { input_ids, attention_mask, ...features });
    }
}
