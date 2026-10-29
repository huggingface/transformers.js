import { Qwen2VLImageProcessor } from '../qwen2_vl/image_processing_qwen2_vl.js';
import { cat } from '../../utils/tensor.js';

export class Lfm3VlImageProcessor extends Qwen2VLImageProcessor {
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
