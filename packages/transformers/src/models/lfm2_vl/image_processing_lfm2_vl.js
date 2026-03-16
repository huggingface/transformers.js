import { ImageProcessor, smart_resize } from '../../image_processors_utils.js';
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
        const ratio_diff = Math.abs(aspect_ratio - ratio[0] / ratio[1]);
        if (ratio_diff < best_ratio_diff) {
            best_ratio_diff = ratio_diff;
            best_ratio = ratio;
        } else if (ratio_diff === best_ratio_diff && area > 0.5 * image_size * image_size * ratio[0] * ratio[1]) {
            best_ratio = ratio;
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
    return ratios.sort((a, b) => a[0] * a[1] - b[0] * b[1]);
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
    const [B, C, H, W] = images.dims;
    const ph = Math.floor(H / patch_size),
        pw = Math.floor(W / patch_size);
    const patch_dim = patch_size * patch_size * C;
    const data = /** @type {Float32Array} */ (images.data);
    const result = new Float32Array(B * ph * pw * patch_dim);
    const ch_stride = H * W;

    for (let b = 0; b < B; ++b) {
        const b_src = b * C * ch_stride;
        const b_dst = b * ph * pw * patch_dim;
        for (let py = 0; py < ph; ++py) {
            for (let px = 0; px < pw; ++px) {
                let off = b_dst + (py * pw + px) * patch_dim;
                for (let dy = 0; dy < patch_size; ++dy) {
                    const row = (py * patch_size + dy) * W + px * patch_size;
                    for (let dx = 0; dx < patch_size; ++dx) {
                        const pixel = row + dx;
                        for (let c = 0; c < C; ++c) {
                            result[off++] = data[b_src + c * ch_stride + pixel];
                        }
                    }
                }
            }
        }
    }

    return new Tensor('float32', result, [B, ph * pw, patch_dim]);
}

/**
 * Pad patches along the patch dimension to `target_length`.
 * @param {Tensor} patches Shape: [1, current_length, patch_dim]
 * @param {number} target_length
 * @returns {{ padded: Tensor, mask: Tensor }}
 */
function pad_along_first_dim(patches, target_length) {
    const [, len, dim] = patches.dims;
    const mask_data = new BigInt64Array(target_length);
    mask_data.fill(1n, 0, len);

    let padded = patches;
    if (len < target_length) {
        const padded_data = new Float32Array(target_length * dim);
        padded_data.set(/** @type {Float32Array} */ (patches.data));
        padded = new Tensor('float32', padded_data, [1, target_length, dim]);
    }

    return { padded, mask: new Tensor('int64', mask_data, [target_length]) };
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
    async _call(images, { return_row_col_info = null } = {}) {
        /** @type {RawImage[][]} */
        let batched_images;
        if (!Array.isArray(images)) {
            batched_images = [[images]];
        } else if (!Array.isArray(images[0])) {
            batched_images = [/** @type {RawImage[]} */ (images)];
        } else {
            batched_images = /** @type {RawImage[][]} */ (images);
        }

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
                const [, height, width] = pixel_values.dims;
                const img = pixel_values.unsqueeze_(0);

                const total_factor = this.encoder_patch_size * this.downsample_factor;
                const f2 = total_factor ** 2;
                const [new_width, new_height] = smart_resize(
                    Math.max(total_factor, height),
                    Math.max(total_factor, width),
                    total_factor,
                    this.min_image_tokens * f2,
                    this.max_image_tokens * f2,
                ).map((x) => Math.max(total_factor, x));

                /** @type {Tensor[]} */
                let tiles;
                let num_rows = 1,
                    num_cols = 1;

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

        if (return_row_col_info ?? this.return_row_col_info) {
            result.image_rows = all_rows;
            result.image_cols = all_cols;
            result.image_sizes = all_image_sizes;
        }

        return result;
    }
}
