import { Qwen2VLImageProcessor } from '../qwen2_vl/image_processing_qwen2_vl.js';
import { smart_resize } from '../../image_processors_utils.js';

export class Glm46VImageProcessor extends Qwen2VLImageProcessor {
    /** @type {Qwen2VLImageProcessor['get_resize_output_image_size']} */
    get_resize_output_image_size(image, size) {
        const factor = this.patch_size * this.merge_size;
        // @ts-expect-error ts(2339)
        const temporal_factor = this.config.temporal_patch_size ?? 2;
        return smart_resize(image.height, image.width, factor, this.min_pixels, this.max_pixels, temporal_factor);
    }
}
