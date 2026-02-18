import { PreTrainedModel, cumsum_masked_fill, default_merge_input_ids_with_image_features } from '../modeling_utils.js';
import { sessionRun } from '../session.js';
import { stack, Tensor, ones_like, zeros } from '../../utils/tensor.js';
import { max } from '../../utils/maths.js';

export class Qwen2VLPreTrainedModel extends PreTrainedModel {
    forward_params = [
        // Text inputs
        'input_ids',
        'attention_mask',
        'position_ids',
        'past_key_values',

        // Vision inputs
        'pixel_values',
        'image_grid_thw',
    ];
}
export class Qwen2VLForConditionalGeneration extends Qwen2VLPreTrainedModel {
    /**
     * Calculate the 3D rope index based on image and video's temporal, height and width in LLM.
     *
     * Explanation:
     *     Each embedding sequence contains vision embedding and text embedding or just contains text embedding.
     *
     *     For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
     *     Examples:
     *         input_ids: [T T T T T], here T is for text.
     *         temporal position_ids: [0, 1, 2, 3, 4]
     *         height position_ids: [0, 1, 2, 3, 4]
     *         width position_ids: [0, 1, 2, 3, 4]
     *
     *     For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
     *     and 1D rotary position embeddin for text part.
     *     Examples:
     *         Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
     *         input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
     *         vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
     *         vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
     *         vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
     *         text temporal position_ids: [3, 4, 5, 6, 7]
     *         text height position_ids: [3, 4, 5, 6, 7]
     *         text width position_ids: [3, 4, 5, 6, 7]
     *         Here we calculate the text start position_ids as the max vision position_ids plus 1.
     *
     * @param {Tensor} input_ids Indices of input sequence tokens in the vocabulary. Tensor of shape `(batch_size, sequence_length)`.
     * @param {Tensor} image_grid_thw (Optional) The temporal, height and width of feature shape of each image in LLM. Tensor of shape `(num_images, 3)`.
     * @param {Tensor} video_grid_thw (Optional) The temporal, height and width of feature shape of each video in LLM. Tensor of shape `(num_videos, 3)`.
     * @param {Tensor} attention_mask (Optional) Mask to avoid performing attention on padding token indices. Tensor of shape `(batch_size, sequence_length)`. Mask values selected in `[0, 1]`:
     * - 1 for tokens that are **not masked**,
     * - 0 for tokens that are **masked**.
     * @returns {[Tensor, Tensor]} [position_ids, mrope_position_deltas] with:
     * - position_ids: Tensor of shape `(3, batch_size, sequence_length)`.
     * - mrope_position_deltas: Tensor of shape `(batch_size)`.
     */
    get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask) {
        // @ts-ignore
        const { vision_config, image_token_id, video_token_id, vision_start_token_id } = this.config;
        const spatial_merge_size = vision_config.spatial_merge_size ?? 2;

        const mrope_position_deltas = [];
        if (image_grid_thw || video_grid_thw) {
            let total_input_ids = input_ids.tolist();
            if (!attention_mask) {
                attention_mask = ones_like(input_ids);
            }

            const attention_mask_list = attention_mask.tolist();
            const position_ids_list = Array.from({ length: 3 }, (_) =>
                Array.from({ length: input_ids.dims[0] }, (_) => Array.from({ length: input_ids.dims[1] }, (_) => 1)),
            );

            const image_grid_thw_list = image_grid_thw ? image_grid_thw.tolist() : [];
            const video_grid_thw_list = video_grid_thw ? video_grid_thw.tolist() : [];

            let image_index = 0;
            let video_index = 0;
            for (let i = 0; i < total_input_ids.length; ++i) {
                const ids = total_input_ids[i].filter((_, j) => attention_mask_list[i][j] == 1);

                const vision_start_indices = ids.reduce((acc, x, idx) => {
                    if (x == vision_start_token_id) acc.push(idx);
                    return acc;
                }, []);

                const vision_tokens = vision_start_indices.map((x) => ids[x + 1]);
                const image_nums = vision_tokens.filter((x) => x == image_token_id).length;
                const video_nums = vision_tokens.filter((x) => x == video_token_id).length;

                /** @type {number[][]} */
                let llm_pos_ids_list = [];
                let st = 0;
                let remain_images = image_nums;
                let remain_videos = video_nums;
                for (let j = 0; j < vision_tokens.length; ++j) {
                    const next_image_token = ids.findIndex((x, i) => i > st && x == image_token_id);
                    const next_video_token = ids.findIndex((x, i) => i > st && x == video_token_id);

                    const ed_image = remain_images > 0 && next_image_token !== -1 ? next_image_token : ids.length + 1;

                    const ed_video = remain_videos > 0 && next_video_token !== -1 ? next_video_token : ids.length + 1;

                    let ed;
                    let t, h, w;
                    if (ed_image < ed_video) {
                        [t, h, w] = image_grid_thw_list[image_index];
                        ++image_index;
                        --remain_images;
                        ed = ed_image;
                    } else {
                        [t, h, w] = video_grid_thw_list[video_index];
                        ++video_index;
                        --remain_videos;
                        ed = ed_video;
                    }

                    const [llm_grid_t, llm_grid_h, llm_grid_w] = [
                        Number(t),
                        Math.floor(Number(h) / spatial_merge_size),
                        Math.floor(Number(w) / spatial_merge_size),
                    ];
                    const text_len = ed - st;
                    const st_idx = llm_pos_ids_list.length > 0 ? max(llm_pos_ids_list.at(-1))[0] + 1 : 0;

                    llm_pos_ids_list.push(Array.from({ length: 3 * text_len }, (_, i) => st_idx + (i % text_len)));

                    const offset = text_len + st_idx;
                    const grid_size = llm_grid_t * llm_grid_h * llm_grid_w;
                    const t_index = Array.from(
                        { length: grid_size },
                        (_, i) => offset + Math.floor(i / (llm_grid_h * llm_grid_w)),
                    );
                    const h_index = Array.from(
                        { length: grid_size },
                        (_, i) => offset + (Math.floor(i / llm_grid_w) % llm_grid_h),
                    );
                    const w_index = Array.from({ length: grid_size }, (_, i) => offset + (i % llm_grid_w));

                    llm_pos_ids_list.push([t_index, h_index, w_index].flat());

                    st = ed + grid_size;
                }

                if (st < ids.length) {
                    const st_idx = llm_pos_ids_list.length > 0 ? max(llm_pos_ids_list.at(-1))[0] + 1 : 0;
                    const text_len = ids.length - st;

                    llm_pos_ids_list.push(Array.from({ length: 3 * text_len }, (_, i) => st_idx + (i % text_len)));
                }

                // NOTE: Each item in llm_pos_ids_list is an array of shape (3, text_len),
                // meaning to perform concatenation along dim=1, we can do the following:
                const num_items = llm_pos_ids_list.reduce((acc, x) => acc + x.length, 0);
                /** @type {number[]} */
                const llm_positions = new Array(num_items);
                let index = 0;
                for (let x = 0; x < 3; ++x) {
                    for (let y = 0; y < llm_pos_ids_list.length; ++y) {
                        const val = llm_pos_ids_list[y];
                        const text_len = val.length / 3;
                        for (let z = x * text_len; z < (x + 1) * text_len; ++z) {
                            llm_positions[index++] = val[z];
                        }
                    }
                }

                let count = 0;
                const attn_mask = attention_mask_list[i];
                for (let y = 0; y < attn_mask.length; ++y) {
                    if (attn_mask[y] == 1) {
                        for (let x = 0; x < 3; ++x) {
                            position_ids_list[x][i][y] = llm_positions[(x * num_items) / 3 + count];
                        }
                        ++count;
                    }
                }

                const max_llm_positions = max(llm_positions)[0];
                mrope_position_deltas.push(max_llm_positions + 1 - total_input_ids[i].length);
            }

            return [
                new Tensor('int64', position_ids_list.flat(Infinity), [3, input_ids.dims[0], input_ids.dims[1]]),
                new Tensor('int64', mrope_position_deltas, [mrope_position_deltas.length, 1]),
            ];
        } else {
            // Text-only
            if (attention_mask) {
                const { data, dims } = cumsum_masked_fill(attention_mask);

                const position_ids = BigInt64Array.from({ length: 3 * data.length }, (_, i) => data[i % data.length]);
                /** @type {bigint[]} */
                const mrope_position_deltas = Array.from(
                    { length: dims[0] },
                    (_, i) => max(data.subarray(dims[1] * i, dims[1] * (i + 1)))[0] + 1n + BigInt(dims[1]),
                );

                return [
                    new Tensor('int64', position_ids, [3, ...dims]),
                    new Tensor('int64', mrope_position_deltas, [mrope_position_deltas.length, 1]),
                ];
            } else {
                const [batch_size, seq_length] = input_ids.dims;
                const position_ids = BigInt64Array.from({ length: 3 * batch_size * seq_length }, (_, i) =>
                    BigInt(Math.floor((i % seq_length) / batch_size)),
                );

                return [new Tensor('int64', position_ids, [3, ...input_ids.dims]), zeros([batch_size, 1])];
            }
        }
    }

    async encode_image({ pixel_values, image_grid_thw }) {
        const features = (await sessionRun(this.sessions['vision_encoder'], { pixel_values, grid_thw: image_grid_thw }))
            .image_features;
        return features;
    }

    _merge_input_ids_with_image_features(kwargs) {
        return default_merge_input_ids_with_image_features({
            // @ts-ignore
            image_token_id: this.config.image_token_id,
            ...kwargs,
        });
    }

    prepare_inputs_for_generation(input_ids, model_inputs, generation_config) {
        // Overwritten -- in specific circumstances we don't want to forward image inputs to the model
        if (model_inputs.attention_mask && !model_inputs.position_ids) {
            // Calculate position_ids and rope_deltas
            if (!model_inputs.past_key_values) {
                [model_inputs.position_ids, model_inputs.rope_deltas] = this.get_rope_index(
                    model_inputs.input_ids,
                    model_inputs.image_grid_thw,
                    model_inputs.video_grid_thw,
                    model_inputs.attention_mask,
                );
            } else {
                model_inputs.pixel_values = null;
                // model_inputs.pixel_values_videos = null;

                const delta = BigInt(Object.values(model_inputs.past_key_values)[0].dims.at(-2));
                const rope_deltas_list = model_inputs.rope_deltas.map((x) => delta + x);
                model_inputs.position_ids = stack([rope_deltas_list, rope_deltas_list, rope_deltas_list], 0);
            }
        }

        return model_inputs;
    }
}
