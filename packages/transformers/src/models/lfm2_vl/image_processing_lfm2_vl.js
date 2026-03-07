import { ImageProcessor } from '../../image_processors_utils.js';
import { Tensor } from '../../utils/tensor.js';
import { cat, interpolate_4d, stack } from '../../utils/tensor.js';

/**
 * Returns the closest integer to `number` that is divisible by `factor`.
 */
function round_by_factor(number, factor) {
    return Math.round(number / factor) * factor;
}

/**
 * Find the closest aspect ratio from target_ratios to match the input aspect ratio.
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
            const target_area = image_size * image_size * ratio[0] * ratio[1];
            if (area > 0.5 * target_area) {
                best_ratio = ratio;
            }
        }
    }
    return best_ratio;
}

/**
 * Compute target ratios for grid layouts.
 */
function get_target_ratios(min_tiles, max_tiles) {
    const ratios = new Set();
    for (let n = min_tiles; n <= max_tiles; ++n) {
        for (let w = 1; w <= n; ++w) {
            for (let h = 1; h <= n; ++h) {
                if (w * h >= min_tiles && w * h <= max_tiles) {
                    ratios.add(`${w},${h}`);
                }
            }
        }
    }
    return Array.from(ratios)
        .map(s => s.split(',').map(Number))
        .sort((a, b) => a[0] * a[1] - b[0] * b[1]);
}

/**
 * Smart resize to ensure dimensions are divisible by encoder_patch_size * downsample_factor
 * while keeping pixel count within [min_pixels, max_pixels].
 */
function smart_resize(height, width, downsample_factor, min_image_tokens, max_image_tokens, encoder_patch_size) {
    const total_factor = encoder_patch_size * downsample_factor;
    const min_pixels = min_image_tokens * encoder_patch_size ** 2 * downsample_factor ** 2;
    const max_pixels = max_image_tokens * encoder_patch_size ** 2 * downsample_factor ** 2;

    let h_bar = Math.max(total_factor, round_by_factor(height, total_factor));
    let w_bar = Math.max(total_factor, round_by_factor(width, total_factor));

    if (h_bar * w_bar > max_pixels) {
        const beta = Math.sqrt((height * width) / max_pixels);
        h_bar = Math.max(total_factor, Math.floor(height / beta / total_factor) * total_factor);
        w_bar = Math.max(total_factor, Math.floor(width / beta / total_factor) * total_factor);
    } else if (h_bar * w_bar < min_pixels) {
        const beta = Math.sqrt(min_pixels / (height * width));
        h_bar = Math.ceil(height * beta / total_factor) * total_factor;
        w_bar = Math.ceil(width * beta / total_factor) * total_factor;
    }

    return [w_bar, h_bar]; // [width, height]
}

/**
 * Convert image tensor to flattened patches.
 * Input shape: [batch, channels, height, width]
 * Output shape: [batch, num_patches, patch_size * patch_size * channels]
 */
function convert_image_to_patches(images, patch_size) {
    const [batch_size, num_channels, image_height, image_width] = images.dims;
    const num_patches_height = Math.floor(image_height / patch_size);
    const num_patches_width = Math.floor(image_width / patch_size);
    const num_patches = num_patches_height * num_patches_width;
    const patch_dim = patch_size * patch_size * num_channels;

    const data = images.data;
    const result = new Float32Array(batch_size * num_patches * patch_dim);

    for (let b = 0; b < batch_size; ++b) {
        for (let ph = 0; ph < num_patches_height; ++ph) {
            for (let pw = 0; pw < num_patches_width; ++pw) {
                const patch_idx = b * num_patches + ph * num_patches_width + pw;
                let offset = 0;
                for (let py = 0; py < patch_size; ++py) {
                    for (let px = 0; px < patch_size; ++px) {
                        const y = ph * patch_size + py;
                        const x = pw * patch_size + px;
                        for (let c = 0; c < num_channels; ++c) {
                            result[patch_idx * patch_dim + offset] =
                                data[b * num_channels * image_height * image_width + c * image_height * image_width + y * image_width + x];
                            offset++;
                        }
                    }
                }
            }
        }
    }

    return new Tensor('float32', result, [batch_size, num_patches, patch_dim]);
}

/**
 * Pad patches along the first (patch) dimension to target_length.
 * Input shape: [batch, current_length, patch_dim]
 * Returns: { padded, mask } where mask is [target_length] int32
 */
function pad_along_first_dim(patches, target_length) {
    const [batch_size, current_length, patch_dim] = patches.dims;
    const padding_length = target_length - current_length;

    const mask_data = new BigInt64Array(target_length).fill(1n);

    if (padding_length > 0) {
        const padded_data = new Float32Array(batch_size * target_length * patch_dim);
        const src = patches.data;

        for (let b = 0; b < batch_size; ++b) {
            const src_offset = b * current_length * patch_dim;
            const dst_offset = b * target_length * patch_dim;
            padded_data.set(src.subarray(src_offset, src_offset + current_length * patch_dim), dst_offset);
            // Remaining is zero-filled by default
        }

        for (let i = current_length; i < target_length; ++i) {
            mask_data[i] = 0n;
        }

        return {
            padded: new Tensor('float32', padded_data, [batch_size, target_length, patch_dim]),
            mask: new Tensor('int64', mask_data, [target_length]),
        };
    }

    return {
        padded: patches,
        mask: new Tensor('int64', mask_data, [target_length]),
    };
}

export class Lfm2VlImageProcessor extends ImageProcessor {
    constructor(config) {
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
     */
    _is_image_too_large(height, width) {
        const total_factor = this.encoder_patch_size * this.downsample_factor;
        const h_bar = Math.max(this.encoder_patch_size, round_by_factor(height, total_factor));
        const w_bar = Math.max(this.encoder_patch_size, round_by_factor(width, total_factor));
        return h_bar * w_bar > this.max_image_tokens * this.encoder_patch_size ** 2 * this.downsample_factor ** 2 * this.max_pixels_tolerance;
    }

    /**
     * Get the grid layout for tiling a large image.
     */
    _get_grid_layout(height, width) {
        const aspect_ratio = width / height;
        const target_ratios = get_target_ratios(this.min_tiles, this.max_tiles);
        const [grid_width, grid_height] = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, width, height, this.tile_size,
        );
        return {
            grid_width,
            grid_height,
            target_width: this.tile_size * grid_width,
            target_height: this.tile_size * grid_height,
        };
    }

    /**
     * @param {import('../../utils/image.js').RawImage|import('../../utils/image.js').RawImage[]|import('../../utils/image.js').RawImage[][]} images
     */
    async _call(images, kwargs = {}) {
        if (!Array.isArray(images)) {
            images = [[images]];
        } else if (!Array.isArray(images[0])) {
            images = [/** @type {import('../../utils/image.js').RawImage[]} */ (images)];
        }

        const return_row_col_info = kwargs.return_row_col_info ?? this.return_row_col_info;

        const all_pixel_values = [];
        const all_pixel_masks = [];
        const all_spatial_shapes = [];
        const all_rows = [];
        const all_cols = [];
        const all_image_sizes = [];

        for (const image_batch of images) {
            // Preprocess each image (resize disabled, rescale+normalize done)
            const preprocessed = await Promise.all(
                image_batch.map(x => this.preprocess(x, { do_pad: false })),
            );

            for (const { pixel_values } of preprocessed) {
                // pixel_values is [C, H, W], unsqueeze to [1, C, H, W]
                const img = pixel_values.unsqueeze_(0);
                const [, , height, width] = img.dims;

                const is_large = this._is_image_too_large(height, width);
                const do_splitting = this.do_image_splitting && !(this.min_tiles === 1 && this.max_tiles === 1);

                // Smart resize for the thumbnail / single-tile size
                const [new_width, new_height] = smart_resize(
                    height, width,
                    this.downsample_factor,
                    this.min_image_tokens,
                    this.max_image_tokens,
                    this.encoder_patch_size,
                );

                /** @type {import('../../utils/tensor.js').Tensor[]} */
                let tiles;
                let num_rows, num_cols;

                if (is_large && do_splitting) {
                    // Split into grid of tiles + optional thumbnail
                    const { grid_width, grid_height, target_width, target_height } =
                        this._get_grid_layout(height, width);
                    num_rows = grid_height;
                    num_cols = grid_width;

                    // Resize to target grid size
                    const resized = await interpolate_4d(img, {
                        size: [target_height, target_width],
                    });

                    // Extract tiles from grid
                    tiles = [];
                    const tile_h = this.tile_size;
                    const tile_w = this.tile_size;
                    for (let r = 0; r < grid_height; ++r) {
                        for (let c = 0; c < grid_width; ++c) {
                            const start_y = r * tile_h;
                            const start_x = c * tile_w;
                            const tile = resized.slice(null, null, [start_y, start_y + tile_h], [start_x, start_x + tile_w]);
                            tiles.push(tile);
                        }
                    }

                    // Add thumbnail if grid has more than 1 tile
                    if (this.use_thumbnail && grid_width * grid_height !== 1) {
                        const thumbnail = await interpolate_4d(img, {
                            size: [new_height, new_width],
                        });
                        tiles.push(thumbnail);
                    }
                } else {
                    // Single tile: just resize
                    num_rows = 1;
                    num_cols = 1;
                    const resized = await interpolate_4d(img, {
                        size: [new_height, new_width],
                    });
                    tiles = [resized];
                }

                // Process each tile: convert to patches, pad, create masks
                for (const tile of tiles) {
                    const [, , th, tw] = tile.dims;
                    const num_patches_h = Math.floor(th / this.encoder_patch_size);
                    const num_patches_w = Math.floor(tw / this.encoder_patch_size);

                    // Convert to patches: [1, num_patches, patch_dim]
                    const patches = convert_image_to_patches(tile, this.encoder_patch_size);

                    // Pad to max_num_patches
                    const { padded, mask } = pad_along_first_dim(patches, this.max_num_patches);

                    all_pixel_values.push(padded);
                    all_pixel_masks.push(mask);
                    all_spatial_shapes.push([num_patches_h, num_patches_w]);
                }

                all_rows.push(num_rows);
                all_cols.push(num_cols);
                all_image_sizes.push([new_height, new_width]);
            }
        }

        // Stack all tiles along batch dimension
        const pixel_values = cat(all_pixel_values, 0);
        const pixel_attention_mask = stack(all_pixel_masks, 0);
        const spatial_shapes = new Tensor(
            'int64',
            BigInt64Array.from(all_spatial_shapes.flat().map(BigInt)),
            [all_spatial_shapes.length, 2],
        );

        const result = {
            pixel_values,
            pixel_attention_mask,
            spatial_shapes,
        };

        if (return_row_col_info) {
            result.image_rows = all_rows;
            result.image_cols = all_cols;
            result.image_sizes = all_image_sizes;
        }

        return result;
    }
}
