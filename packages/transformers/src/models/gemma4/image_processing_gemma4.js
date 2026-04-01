import { Callable } from '../../utils/generic.js';
import { Tensor, stack } from '../../utils/tensor.js';
import { RawImage } from '../../utils/image.js';

/**
 * Compute the target size for aspect-ratio-preserving resize.
 *
 * @param {number} height Original image height.
 * @param {number} width Original image width.
 * @param {number} patch_size Size of each patch in pixels.
 * @param {number} max_patches Maximum number of patches allowed.
 * @param {number} pooling_kernel_size Spatial pooling kernel size.
 * @returns {[number, number]} Target [height, width].
 */
function get_aspect_ratio_preserving_size(height, width, patch_size, max_patches, pooling_kernel_size) {
    const target_px = max_patches * patch_size ** 2;
    const factor = Math.sqrt(target_px / (height * width));
    const side_mult = pooling_kernel_size * patch_size;

    let target_height = Math.floor((factor * height) / side_mult) * side_mult;
    let target_width = Math.floor((factor * width) / side_mult) * side_mult;

    if (target_height === 0 && target_width === 0) {
        throw new Error(
            `Attempting to resize to a 0 x 0 image. Resized height should be divisible by ` +
                `\`pooling_kernel_size * patch_size\`=${side_mult}.`,
        );
    }

    const max_side_length = Math.floor(max_patches / pooling_kernel_size ** 2) * side_mult;
    if (target_height === 0) {
        target_height = side_mult;
        target_width = Math.min(Math.floor(width / height) * side_mult, max_side_length);
    } else if (target_width === 0) {
        target_width = side_mult;
        target_height = Math.min(Math.floor(height / width) * side_mult, max_side_length);
    }

    return [target_height, target_width];
}

/**
 * Convert HWC image data to patches and position IDs, then pad.
 *
 * Patchification produces the same layout as Python's:
 *   CHW.reshape(C, pH, ps, pW, ps).transpose(1, 3, 2, 4, 0).reshape(pH*pW, ps*ps*C)
 * The transpose yields (pH, pW, ps_h, ps_w, C), so within each patch the
 * flattened order is (dy, dx, c) — which is exactly HWC order.
 * This lets us read directly from HWC source data without a CHW conversion.
 *
 * @param {Float32Array} hwc_data Image pixel data in HWC layout.
 * @param {number} H Image height.
 * @param {number} W Image width.
 * @param {number} C Number of channels.
 * @param {number} patch_size Patch size in pixels.
 * @param {number} max_patches Maximum patches (for padding).
 * @param {number} pooling_kernel_size Pooling kernel size (for soft token count).
 * @returns {{ patches: Tensor, positions: Tensor, num_soft_tokens: number }}
 */
function patchify(hwc_data, H, W, C, patch_size, max_patches, pooling_kernel_size) {
    const num_patches_h = Math.floor(H / patch_size);
    const num_patches_w = Math.floor(W / patch_size);
    const num_patches = num_patches_h * num_patches_w;
    const patch_dim = patch_size * patch_size * C;

    // Patchify: iterate (pH, pW, dy, dx, c) — reads HWC in order
    const patch_data = new Float32Array(max_patches * patch_dim); // zero-padded
    let out = 0;
    for (let ph = 0; ph < num_patches_h; ++ph) {
        for (let pw = 0; pw < num_patches_w; ++pw) {
            for (let dy = 0; dy < patch_size; ++dy) {
                const row_offset = (ph * patch_size + dy) * W * C + pw * patch_size * C;
                for (let dx = 0; dx < patch_size; ++dx) {
                    const src = row_offset + dx * C;
                    for (let c = 0; c < C; ++c) {
                        patch_data[out++] = hwc_data[src + c];
                    }
                }
            }
        }
    }

    // Position IDs: meshgrid(arange(W), arange(H), indexing="xy") → [col, row]
    const pos_data = new BigInt64Array(max_patches * 2).fill(-1n); // -1 padding
    let idx = 0;
    for (let row = 0; row < num_patches_h; ++row) {
        for (let col = 0; col < num_patches_w; ++col) {
            pos_data[idx++] = BigInt(col);
            pos_data[idx++] = BigInt(row);
        }
    }

    return {
        patches: new Tensor('float32', patch_data, [max_patches, patch_dim]),
        positions: new Tensor('int64', pos_data, [max_patches, 2]),
        num_soft_tokens: Math.floor(num_patches / pooling_kernel_size ** 2),
    };
}

export class Gemma4ImageProcessor extends Callable {
    /** @param {Record<string, any>} config */
    constructor(config) {
        super();
        this.config = config;
        this.patch_size = config.patch_size ?? 16;
        this.max_soft_tokens = config.max_soft_tokens ?? 280;
        this.pooling_kernel_size = config.pooling_kernel_size ?? 3;
        this.resample = config.resample ?? 3; // bicubic
        this.rescale_factor = config.rescale_factor ?? 1 / 255;
        this.do_rescale = config.do_rescale ?? true;
        this.do_resize = config.do_resize ?? true;
        this.do_convert_rgb = config.do_convert_rgb ?? true;
    }

    /**
     * @param {RawImage|RawImage[]} images
     * @returns {Promise<{ pixel_values: Tensor, image_position_ids: Tensor, num_soft_tokens_per_image: number[] }>}
     */
    async _call(images) {
        if (!Array.isArray(images)) {
            images = [images];
        }

        const { patch_size, pooling_kernel_size } = this;
        const max_patches = this.max_soft_tokens * pooling_kernel_size ** 2;

        const all_patches = [];
        const all_positions = [];
        const num_soft_tokens_per_image = [];

        for (let image of images) {
            if (this.do_convert_rgb) {
                image = image.rgb();
            }

            if (this.do_resize) {
                const [target_h, target_w] = get_aspect_ratio_preserving_size(
                    image.height,
                    image.width,
                    patch_size,
                    max_patches,
                    pooling_kernel_size,
                );
                if (target_h !== image.height || target_w !== image.width) {
                    image = await image.resize(target_w, target_h, { resample: this.resample });
                }
            }

            // Rescale in-place
            const pixelData = Float32Array.from(image.data);
            if (this.do_rescale) {
                for (let i = 0; i < pixelData.length; ++i) {
                    pixelData[i] *= this.rescale_factor;
                }
            }

            const { patches, positions, num_soft_tokens } = patchify(
                pixelData,
                image.height,
                image.width,
                image.channels,
                patch_size,
                max_patches,
                pooling_kernel_size,
            );

            all_patches.push(patches);
            all_positions.push(positions);
            num_soft_tokens_per_image.push(num_soft_tokens);
        }

        return {
            pixel_values: stack(all_patches, 0),
            image_position_ids: stack(all_positions, 0),
            num_soft_tokens_per_image,
        };
    }
}
