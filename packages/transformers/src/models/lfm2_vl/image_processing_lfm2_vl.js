import { ImageProcessor } from '../../image_processors_utils.js';
import { Tensor, cat, interpolate_4d, stack } from '../../utils/tensor.js';

/**
 * @typedef {import('../../utils/image.js').RawImage} RawImage
 */

/**
 * Returns the closest integer to `number` that is divisible by `factor`.
 * @param {number} number
 * @param {number} factor
 * @returns {number}
 */
function round_by_factor(number, factor) {
    return Math.round(number / factor) * factor;
}

/**
 * Find the closest aspect ratio from target_ratios to match the input aspect ratio.
 * @param {number} aspect_ratio
 * @param {number[][]} target_ratios
 * @param {number} width
 * @param {number} height
 * @param {number} image_size
 * @returns {number[]}
 */
function find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size) {
    let best_ratio_diff = Infinity;
    let best_ratio = [1, 1];
    const area = width * height;

    for (const ratio of target_ratios) {
        const target_aspect_ratio = ratio[0] / ratio[1];
        const ratio_diff = Math.abs(aspect_ratio - target_aspect_ratio);

        if (ratio_diff < best_ratio_diff) {
            best_ratio_diff = ratio_diff;
            best_ratio = ratio;
        } else if (ratio_diff === best_ratio_diff) {
            if (area > 0.5 * image_size * image_size * ratio[0] * ratio[1]) {
                best_ratio = ratio;
            }
        }
    }
    return best_ratio;
}

/**
 * Compute all valid (width, height) tile ratios for the given range.
 * @param {number} min_tiles
 * @param {number} max_tiles
 * @returns {number[][]}
 */
function get_target_ratios(min_tiles, max_tiles) {
    /** @type {number[][]} */
    const ratios = [];
    const seen = new Set();
    for (let n = min_tiles; n <= max_tiles; ++n) {
        for (let w = 1; w <= n; ++w) {
            for (let h = 1; h <= n; ++h) {
                const product = w * h;
                if (product >= min_tiles && product <= max_tiles) {
                    const key = (w << 16) | h;
                    if (!seen.has(key)) {
                        seen.add(key);
                        ratios.push([w, h]);
                    }
                }
            }
        }
    }
    ratios.sort((a, b) => a[0] * a[1] - b[0] * b[1]);
    return ratios;
}

/**
 * Smart resize to ensure dimensions are divisible by `encoder_patch_size * downsample_factor`
 * while keeping pixel count within the allowed range.
 * @param {number} height
 * @param {number} width
 * @param {number} downsample_factor
 * @param {number} min_image_tokens
 * @param {number} max_image_tokens
 * @param {number} encoder_patch_size
 * @returns {[number, number]} [width, height]
 */
function smart_resize(height, width, downsample_factor, min_image_tokens, max_image_tokens, encoder_patch_size) {
    const total_factor = encoder_patch_size * downsample_factor;
    const min_pixels = min_image_tokens * (encoder_patch_size * downsample_factor) ** 2;
    const max_pixels = max_image_tokens * (encoder_patch_size * downsample_factor) ** 2;

    let h_bar = Math.max(total_factor, round_by_factor(height, total_factor));
    let w_bar = Math.max(total_factor, round_by_factor(width, total_factor));

    if (h_bar * w_bar > max_pixels) {
        const beta = Math.sqrt((height * width) / max_pixels);
        h_bar = Math.max(total_factor, Math.floor(height / beta / total_factor) * total_factor);
        w_bar = Math.max(total_factor, Math.floor(width / beta / total_factor) * total_factor);
    } else if (h_bar * w_bar < min_pixels) {
        const beta = Math.sqrt(min_pixels / (height * width));
        h_bar = Math.ceil((height * beta) / total_factor) * total_factor;
        w_bar = Math.ceil((width * beta) / total_factor) * total_factor;
    }

    return [w_bar, h_bar];
}

/**
 * Convert image tensor to flattened patches.
 *
 * Equivalent to PyTorch: `images.reshape(B, C, ph, ps, pw, ps).permute(0, 2, 4, 3, 5, 1).reshape(B, ph*pw, -1)`
 * @param {Tensor} images Shape: [batch, channels, height, width]
 * @param {number} patch_size
 * @returns {Tensor} Shape: [batch, num_patches, patch_size * patch_size * channels]
 */
function convert_image_to_patches(images, patch_size) {
    const [batch_size, num_channels, image_height, image_width] = images.dims;
    const num_patches_h = Math.floor(image_height / patch_size);
    const num_patches_w = Math.floor(image_width / patch_size);
    const num_patches = num_patches_h * num_patches_w;
    const patch_dim = patch_size * patch_size * num_channels;

    const data = /** @type {Float32Array} */ (images.data);
    const result = new Float32Array(batch_size * num_patches * patch_dim);

    const channel_stride = image_height * image_width;
    const batch_stride = num_channels * channel_stride;

    for (let b = 0; b < batch_size; ++b) {
        const b_src = b * batch_stride;
        const b_dst = b * num_patches * patch_dim;
        for (let ph = 0; ph < num_patches_h; ++ph) {
            const y_base = ph * patch_size;
            for (let pw = 0; pw < num_patches_w; ++pw) {
                const x_base = pw * patch_size;
                let offset = b_dst + (ph * num_patches_w + pw) * patch_dim;
                for (let py = 0; py < patch_size; ++py) {
                    const row_offset = (y_base + py) * image_width + x_base;
                    for (let px = 0; px < patch_size; ++px) {
                        const pixel_offset = row_offset + px;
                        for (let c = 0; c < num_channels; ++c) {
                            result[offset++] = data[b_src + c * channel_stride + pixel_offset];
                        }
                    }
                }
            }
        }
    }

    return new Tensor('float32', result, [batch_size, num_patches, patch_dim]);
}

/**
 * Pad patches along the patch dimension to `target_length`.
 * @param {Tensor} patches Shape: [1, current_length, patch_dim]
 * @param {number} target_length
 * @returns {{ padded: Tensor, mask: Tensor }}
 */
function pad_along_first_dim(patches, target_length) {
    const current_length = patches.dims[1];
    const patch_dim = patches.dims[2];
    const padding_length = target_length - current_length;

    const mask_data = new BigInt64Array(target_length);
    mask_data.fill(1n, 0, current_length);
    // Remaining is 0n by default

    let padded;
    if (padding_length > 0) {
        const padded_data = new Float32Array(target_length * patch_dim);
        padded_data.set(/** @type {Float32Array} */ (patches.data));
        padded = new Tensor('float32', padded_data, [1, target_length, patch_dim]);
    } else {
        padded = patches;
    }

    return {
        padded,
        mask: new Tensor('int64', mask_data, [target_length]),
    };
}

export class Lfm2VlImageProcessor extends ImageProcessor {
    constructor(/** @type {Record<string, any>} */ config) {
        super(config);

        this.downsample_factor = config.downsample_factor ?? 2;
        this.do_image_splitting = config.do_image_splitting ?? true;
        this.min_tiles = config.min_tiles ?? 2;
        this.max_tiles = config.max_tiles ?? 10;
        this.use_thumbnail = config.use_thumbnail ?? true;
        this.min_image_tokens = config.min_image_tokens ?? 64;
        this.max_image_tokens = config.max_image_tokens ?? 256;
        this.encoder_patch_size = config.encoder_patch_size ?? config.patch_size ?? 16;
        this.tile_size = config.tile_size ?? 512;
        this.max_pixels_tolerance = config.max_pixels_tolerance ?? 2.0;
        this.return_row_col_info = config.return_row_col_info ?? false;

        const max_thumbnail_patches = this.max_image_tokens * this.downsample_factor ** 2;
        const tile_size_patches = this.do_image_splitting ? (this.tile_size / this.encoder_patch_size) ** 2 : 0;
        this.max_num_patches = Math.max(max_thumbnail_patches, tile_size_patches);
    }

    /**
     * Check if the image is too large to be processed as a single tile.
     * @param {number} height
     * @param {number} width
     * @returns {boolean}
     */
    _is_image_too_large(height, width) {
        const total_factor = this.encoder_patch_size * this.downsample_factor;
        const h_bar = Math.max(this.encoder_patch_size, round_by_factor(height, total_factor));
        const w_bar = Math.max(this.encoder_patch_size, round_by_factor(width, total_factor));
        return (
            h_bar * w_bar >
            this.max_image_tokens * (this.encoder_patch_size * this.downsample_factor) ** 2 * this.max_pixels_tolerance
        );
    }

    /**
     * Get the grid layout for tiling a large image.
     * @param {number} height
     * @param {number} width
     * @returns {{ grid_width: number, grid_height: number, target_width: number, target_height: number }}
     */
    _get_grid_layout(height, width) {
        const target_ratios = get_target_ratios(this.min_tiles, this.max_tiles);
        const [grid_width, grid_height] = find_closest_aspect_ratio(
            width / height,
            target_ratios,
            width,
            height,
            this.tile_size,
        );
        return {
            grid_width,
            grid_height,
            target_width: this.tile_size * grid_width,
            target_height: this.tile_size * grid_height,
        };
    }

    /** @param {RawImage|RawImage[]|RawImage[][]} images */
    // @ts-expect-error
    async _call(images, kwargs = {}) {
        /** @type {RawImage[][]} */
        let batched_images;
        if (!Array.isArray(images)) {
            batched_images = [[images]];
        } else if (!Array.isArray(images[0])) {
            batched_images = [/** @type {RawImage[]} */ (images)];
        } else {
            batched_images = /** @type {RawImage[][]} */ (images);
        }

        const return_row_col_info = kwargs.return_row_col_info ?? this.return_row_col_info;

        /** @type {Tensor[]} */
        const all_pixel_values = [];
        /** @type {Tensor[]} */
        const all_pixel_masks = [];
        /** @type {number[][]} */
        const all_spatial_shapes = [];
        /** @type {number[]} */
        const all_rows = [];
        /** @type {number[]} */
        const all_cols = [];
        /** @type {number[][]} */
        const all_image_sizes = [];

        for (const image_batch of batched_images) {
            const preprocessed = await Promise.all(image_batch.map((x) => this.preprocess(x, { do_pad: false })));

            for (const { pixel_values } of preprocessed) {
                const img = pixel_values.unsqueeze_(0);
                const [, , height, width] = img.dims;

                const [new_width, new_height] = smart_resize(
                    height,
                    width,
                    this.downsample_factor,
                    this.min_image_tokens,
                    this.max_image_tokens,
                    this.encoder_patch_size,
                );

                /** @type {Tensor[]} */
                let tiles;
                let num_rows, num_cols;

                const is_large = this._is_image_too_large(height, width);
                const do_splitting = this.do_image_splitting && !(this.min_tiles === 1 && this.max_tiles === 1);

                if (is_large && do_splitting) {
                    const { grid_width, grid_height, target_width, target_height } = this._get_grid_layout(
                        height,
                        width,
                    );
                    num_rows = grid_height;
                    num_cols = grid_width;

                    const resized = await interpolate_4d(img, {
                        size: [target_height, target_width],
                    });

                    tiles = [];
                    for (let r = 0; r < grid_height; ++r) {
                        for (let c = 0; c < grid_width; ++c) {
                            const y = r * this.tile_size;
                            const x = c * this.tile_size;
                            tiles.push(resized.slice(null, null, [y, y + this.tile_size], [x, x + this.tile_size]));
                        }
                    }

                    if (this.use_thumbnail && grid_width * grid_height !== 1) {
                        tiles.push(await interpolate_4d(img, { size: [new_height, new_width] }));
                    }
                } else {
                    num_rows = 1;
                    num_cols = 1;
                    tiles = [await interpolate_4d(img, { size: [new_height, new_width] })];
                }

                for (const tile of tiles) {
                    const [, , th, tw] = tile.dims;
                    const patches = convert_image_to_patches(tile, this.encoder_patch_size);
                    const { padded, mask } = pad_along_first_dim(patches, this.max_num_patches);

                    all_pixel_values.push(padded);
                    all_pixel_masks.push(mask);
                    all_spatial_shapes.push([
                        Math.floor(th / this.encoder_patch_size),
                        Math.floor(tw / this.encoder_patch_size),
                    ]);
                }

                all_rows.push(num_rows);
                all_cols.push(num_cols);
                all_image_sizes.push([new_height, new_width]);
            }
        }

        /** @type {Record<string, any>} */
        const result = {
            pixel_values: cat(all_pixel_values, 0),
            pixel_attention_mask: stack(all_pixel_masks, 0),
            spatial_shapes: new Tensor('int64', BigInt64Array.from(all_spatial_shapes.flat(), BigInt), [
                all_spatial_shapes.length,
                2,
            ]),
        };

        if (return_row_col_info) {
            result.image_rows = all_rows;
            result.image_cols = all_cols;
            result.image_sizes = all_image_sizes;
        }

        return result;
    }
}
