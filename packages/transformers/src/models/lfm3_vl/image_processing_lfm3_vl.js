import { Qwen2VLImageProcessor } from '../qwen2_vl/image_processing_qwen2_vl.js';
import { cat } from '../../utils/tensor.js';

export class Lfm3VlImageProcessor extends Qwen2VLImageProcessor {
    constructor(config) {
        super({
            do_resize: true,
            size: { shortest_edge: 65536, longest_edge: 16777216 },
            resample: 3,
            do_rescale: true,
            rescale_factor: 1 / 255,
            do_normalize: true,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
            do_convert_rgb: true,
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 2,
            ...config,
        });
    }

    async _call(images, ...args) {
        if (!Array.isArray(images)) images = [images];

        // Patchify each image independently: images may have different spatial grids.
        const inputs = await Promise.all(images.map((image) => super._call(image, ...args)));
        return {
            pixel_values: cat(
                inputs.map((x) => x.pixel_values),
                0,
            ),
            image_grid_thw: cat(
                inputs.map((x) => x.image_grid_thw),
                0,
            ),
            original_sizes: inputs.flatMap((x) => x.original_sizes),
            reshaped_input_sizes: inputs.flatMap((x) => x.reshaped_input_sizes),
        };
    }
}
