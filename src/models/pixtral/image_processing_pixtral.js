import {
    ImageProcessor,
} from "../../base/image_processors_utils.js";

export class PixtralImageProcessor extends ImageProcessor {

    /** @type {ImageProcessor['get_resize_output_image_size']} */
    get_resize_output_image_size(image, size) {
        const { longest_edge } = size;
        if (longest_edge === undefined) {
            throw new Error("size must contain 'longest_edge'");
        }

        const [srcWidth, srcHeight] = image.size;

        const ratio = Math.max(srcWidth, srcHeight) / longest_edge;

        let newWidth = srcWidth;
        let newHeight = srcHeight;
        if (ratio > 1) {
            newWidth = Math.floor(srcWidth / ratio);
            newHeight = Math.floor(srcHeight / ratio);
        }

        // @ts-expect-error TS2339
        const { patch_size, spatial_merge_size } = this.config;
        if (!spatial_merge_size) {
            throw new Error("config must contain 'spatial_merge_size'");
        }
        const real_patch_size = patch_size * spatial_merge_size;

        // Calculate number of tokens
        const num_width_tokens = Math.floor((newWidth - 1) / real_patch_size) + 1;
        const num_height_tokens = Math.floor((newHeight - 1) / real_patch_size) + 1;

        return [num_width_tokens * real_patch_size, num_height_tokens * real_patch_size];
    }
}
