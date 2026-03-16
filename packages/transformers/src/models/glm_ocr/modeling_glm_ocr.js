import { Qwen2_5_VLForConditionalGeneration } from '../qwen2_5_vl/modeling_qwen2_5_vl.js';
import { Tensor, ones_like, zeros } from '../../utils/tensor.js';
import { cumsum_masked_fill } from '../modeling_utils.js';
import { max } from '../../utils/maths.js';

export class GlmOcrForConditionalGeneration extends Qwen2_5_VLForConditionalGeneration {
    /**
     * Compute 3D positional indices for vision tokens.
     * Temporal is constant, height is repeat-interleaved, width tiles.
     * @param {number} start_position
     * @param {number[]} grid_thw [T, H, W]
     * @param {number} temp_merge_size
     * @param {number} spatial_merge_size
     * @returns {number[]} Flat array of length 3 * seq_len: [temporal..., height..., width...]
     */
    get_vision_position_ids(start_position, grid_thw, temp_merge_size, spatial_merge_size) {
        const llm_grid_t = Math.floor(grid_thw[0] / temp_merge_size);
        const llm_grid_h = Math.floor(grid_thw[1] / spatial_merge_size);
        const llm_grid_w = Math.floor(grid_thw[2] / spatial_merge_size);
        const seq_len = llm_grid_h * llm_grid_w * llm_grid_t;

        const t_pos = Array.from({ length: seq_len }, () => start_position);
        const h_pos = Array.from(
            { length: seq_len },
            (_, i) => start_position + Math.floor(i / (llm_grid_w * llm_grid_t)),
        );
        const w_pos = Array.from({ length: seq_len }, (_, i) => start_position + (i % llm_grid_w));

        return [...t_pos, ...h_pos, ...w_pos];
    }

    /**
     * GlmOcr uses mm_token_type_ids-style grouping (image tokens identified by image_token_id)
     * instead of vision_start_token_id scanning used by Qwen2VL.
     * After a vision segment, position advances by max(h, w) / spatial_merge_size.
     */
    get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask) {
        // @ts-ignore
        const { vision_config, image_token_id } = this.config;
        const spatial_merge_size = vision_config.spatial_merge_size ?? 2;

        const mrope_position_deltas = [];
        if (image_grid_thw || video_grid_thw) {
            const total_input_ids = input_ids.tolist();
            if (!attention_mask) {
                attention_mask = ones_like(input_ids);
            }
            const attention_mask_list = attention_mask.tolist();
            const position_ids_list = Array.from({ length: 3 }, () =>
                Array.from({ length: input_ids.dims[0] }, () => Array.from({ length: input_ids.dims[1] }, () => 0)),
            );
            const image_grid_thw_list = image_grid_thw ? image_grid_thw.tolist() : [];

            let image_index = 0;
            for (let batch_idx = 0; batch_idx < total_input_ids.length; ++batch_idx) {
                const ids = total_input_ids[batch_idx];
                const attn_mask = attention_mask_list[batch_idx];
                const filtered_ids = ids.filter((_, j) => attn_mask[j] == 1);

                // Build modality groups: 0=text, 1=image (by image_token_id)
                const groups = [];
                let group_start = 0;
                let current_type = filtered_ids[0] == image_token_id ? 1 : 0;
                for (let j = 1; j <= filtered_ids.length; ++j) {
                    const t = j < filtered_ids.length ? (filtered_ids[j] == image_token_id ? 1 : 0) : -1;
                    if (t !== current_type) {
                        groups.push([current_type, group_start, j]);
                        group_start = j;
                        current_type = t;
                    }
                }

                let current_pos = 0;
                /** @type {number[][]} */
                const llm_pos_ids_list = [];

                for (const [modality_type, start_idx, end_idx] of groups) {
                    if (modality_type === 0) {
                        const text_len = end_idx - start_idx;
                        llm_pos_ids_list.push(
                            Array.from({ length: 3 * text_len }, (_, i) => current_pos + (i % text_len)),
                        );
                        current_pos += text_len;
                    } else {
                        const grid_thw = image_grid_thw_list[image_index++].map(Number);
                        const temp_merge_size = grid_thw[0];
                        llm_pos_ids_list.push(
                            this.get_vision_position_ids(current_pos, grid_thw, temp_merge_size, spatial_merge_size),
                        );
                        current_pos += Math.max(grid_thw[1], grid_thw[2]) / spatial_merge_size;
                    }
                }

                // Reorder from per-segment [t,h,w] to global [all_t, all_h, all_w]
                const total_len = llm_pos_ids_list.reduce((acc, x) => acc + x.length, 0);
                const llm_positions = new Array(total_len);
                let index = 0;
                for (let x = 0; x < 3; ++x) {
                    for (const val of llm_pos_ids_list) {
                        const seg_len = val.length / 3;
                        for (let z = x * seg_len; z < (x + 1) * seg_len; ++z) {
                            llm_positions[index++] = val[z];
                        }
                    }
                }

                let count = 0;
                for (let y = 0; y < attn_mask.length; ++y) {
                    if (attn_mask[y] == 1) {
                        for (let x = 0; x < 3; ++x) {
                            position_ids_list[x][batch_idx][y] = llm_positions[(x * total_len) / 3 + count];
                        }
                        ++count;
                    }
                }

                mrope_position_deltas.push(max(llm_positions)[0] + 1 - total_input_ids[batch_idx].length);
            }

            return [
                new Tensor('int64', position_ids_list.flat(Infinity), [3, input_ids.dims[0], input_ids.dims[1]]),
                new Tensor('int64', mrope_position_deltas, [mrope_position_deltas.length, 1]),
            ];
        } else {
            // Text-only: same as Qwen2VL
            if (attention_mask) {
                const { data, dims } = cumsum_masked_fill(attention_mask);
                const position_ids = BigInt64Array.from({ length: 3 * data.length }, (_, i) => data[i % data.length]);
                const delta = Array.from(
                    { length: dims[0] },
                    (_, i) => max(data.subarray(dims[1] * i, dims[1] * (i + 1)))[0] + 1n + BigInt(dims[1]),
                );
                return [new Tensor('int64', position_ids, [3, ...dims]), new Tensor('int64', delta, [delta.length, 1])];
            } else {
                const [batch_size, seq_length] = input_ids.dims;
                const position_ids = BigInt64Array.from({ length: 3 * batch_size * seq_length }, (_, i) =>
                    BigInt(Math.floor((i % seq_length) / batch_size)),
                );
                return [new Tensor('int64', position_ids, [3, ...input_ids.dims]), zeros([batch_size, 1])];
            }
        }
    }
}
