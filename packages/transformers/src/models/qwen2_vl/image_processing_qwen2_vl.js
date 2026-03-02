import { ImageProcessor } from '../../image_processors_utils.js';
import { cat, Tensor } from '../../utils/tensor.js';

/**
 * Rescales the image so that the following conditions are met:
 *
 * 1. Both dimensions (height and width) are divisible by 'factor'.
 * 2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
 * 3. The aspect ratio of the image is maintained as closely as possible.
 *
 * @param {number} height The height of the image.
 * @param {number} width The width of the image.
 * @param {number} [factor=28] The factor to use for resizing.
 * @param {number} [min_pixels=56*56] The minimum number of pixels.
 * @param {number} [max_pixels=14*14*4*1280] The maximum number of pixels.
 * @returns {[number, number]} The new height and width of the image.
 * @throws {Error} If the height or width is smaller than the factor.
 */
function smart_resize(height, width, factor = 28, min_pixels = 56 * 56, max_pixels = 14 * 14 * 4 * 1280) {
    if (height < factor || width < factor) {
        throw new Error(`height:${height} or width:${width} must be larger than factor:${factor}`);
    } else if (Math.max(height, width) / Math.min(height, width) > 200) {
        throw new Error(
            `absolute aspect ratio must be smaller than 200, got ${Math.max(height, width) / Math.min(height, width)}`,
        );
    }

    let h_bar = Math.round(height / factor) * factor;
    let w_bar = Math.round(width / factor) * factor;

    if (h_bar * w_bar > max_pixels) {
        const beta = Math.sqrt((height * width) / max_pixels);
        h_bar = Math.floor(height / beta / factor) * factor;
        w_bar = Math.floor(width / beta / factor) * factor;
    } else if (h_bar * w_bar < min_pixels) {
        const beta = Math.sqrt(min_pixels / (height * width));
        h_bar = Math.ceil((height * beta) / factor) * factor;
        w_bar = Math.ceil((width * beta) / factor) * factor;
    }

    return [h_bar, w_bar];
}

export class Qwen2VLImageProcessor extends ImageProcessor {
    constructor(config) {
        super(config);
        this.min_pixels = config.min_pixels ?? config.size?.shortest_edge;
        this.max_pixels = config.max_pixels ?? config.size?.longest_edge;
        this.patch_size = config.patch_size;
        this.merge_size = config.merge_size;
    }

    /** @type {ImageProcessor['get_resize_output_image_size']} */
    get_resize_output_image_size(image, size) {
        const factor = this.patch_size * this.merge_size;
        return smart_resize(image.height, image.width, factor, this.min_pixels, this.max_pixels);
    }

    async _call(images, ...args) {
        const { pixel_values, original_sizes, reshaped_input_sizes } = await super._call(images, ...args);

        let patches = pixel_values;

        // @ts-ignore
        const { temporal_patch_size, merge_size, patch_size } = this.config;
        if (patches.dims[0] === 1) {
            // Equivalent to np.tile(patches, (self.temporal_patch_size, 1, 1, 1))
            patches = cat(
                Array.from({ length: temporal_patch_size }, () => patches),
                0,
            );
        }

        const grid_t = patches.dims[0] / temporal_patch_size;
        const channel = patches.dims[1];
        const grid_h = Math.floor(patches.dims[2] / patch_size);
        const grid_w = Math.floor(patches.dims[3] / patch_size);

        const flatten_patches = patches
            .view(
                grid_t,
                temporal_patch_size,
                channel,
                Math.floor(grid_h / merge_size),
                merge_size,
                patch_size,
                Math.floor(grid_w / merge_size),
                merge_size,
                patch_size,
            )
            .permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
            .view(grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size);

        const image_grid_thw = new Tensor('int64', [grid_t, grid_h, grid_w], [1, 3]);

        return {
            pixel_values: flatten_patches,
            image_grid_thw,
            original_sizes,
            reshaped_input_sizes,
        };
    }
}
