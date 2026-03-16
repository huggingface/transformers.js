import { Qwen2_5_VLForConditionalGeneration } from '../qwen2_5_vl/modeling_qwen2_5_vl.js';

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
    _get_multimodal_rope_positions({
        filtered_ids,
        image_grid_thw_list,
        video_grid_thw_list,
        spatial_merge_size,
        state,
    }) {
        // @ts-ignore
        const { image_token_id } = this.config;

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
                llm_pos_ids_list.push(Array.from({ length: 3 * text_len }, (_, i) => current_pos + (i % text_len)));
                current_pos += text_len;
            } else {
                const grid_thw = image_grid_thw_list[state.image_index++].map(Number);
                const temp_merge_size = grid_thw[0];
                llm_pos_ids_list.push(
                    this.get_vision_position_ids(current_pos, grid_thw, temp_merge_size, spatial_merge_size),
                );
                current_pos += Math.max(grid_thw[1], grid_thw[2]) / spatial_merge_size;
            }
        }

        return llm_pos_ids_list;
    }
}
