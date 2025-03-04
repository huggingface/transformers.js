import { ImageProcessor } from "../../base/image_processors_utils.js";

/**
 * @typedef {Object} ImageProcessorConfig A configuration object used to create an image processor.
 * @property {function} [progress_callback=null] If specified, this function will be called during model construction, to provide the user with progress updates.
 * @property {number[]} [image_mean] The mean values for image normalization.
 * @property {number[]} [image_std] The standard deviation values for image normalization.
 * @property {boolean} [do_rescale] Whether to rescale the image pixel values to the [0,1] range.
 * @property {number} [rescale_factor] The factor to use for rescaling the image pixel values.
 * @property {boolean} [do_normalize] Whether to normalize the image pixel values.
 * @property {boolean} [do_resize] Whether to resize the image.
 * @property {number} [resample] What method to use for resampling.
 * @property {number|Object} [size] The size to resize the image to.
 * @property {number|Object} [image_size] The size to resize the image to (same as `size`).
 * @property {boolean} [do_flip_channel_order=false] Whether to flip the color channels from RGB to BGR.
 * Can be overridden by the `do_flip_channel_order` parameter in the `preprocess` method.
 * @property {boolean} [do_center_crop] Whether to center crop the image to the specified `crop_size`.
 * Can be overridden by `do_center_crop` in the `preprocess` method.
 * @property {boolean} [do_thumbnail] Whether to resize the image using thumbnail method.
 * @property {boolean} [keep_aspect_ratio] If `true`, the image is resized to the largest possible size such that the aspect ratio is preserved.
 * Can be overridden by `keep_aspect_ratio` in `preprocess`.
 * @property {number} [ensure_multiple_of] If `do_resize` is `true`, the image is resized to a size that is a multiple of this value.
 * Can be overridden by `ensure_multiple_of` in `preprocess`.
 *
 * @property {number[]} [mean] The mean values for image normalization (same as `image_mean`).
 * @property {number[]} [std] The standard deviation values for image normalization (same as `image_std`).
 */
export class U2NetImageProcessor extends ImageProcessor {
    /**
     * Pad the image by a certain amount.
     * @param {Float32Array} pixelData The pixel data to pad.
     * @param {number[]} imgDims The dimensions of the image (height, width, channels).
     * @param {{width:number; height:number}|number|'square'} padSize The dimensions of the padded image.
     * @param {Object} options The options for padding.
     * @param {'constant'|'symmetric'} [options.mode='constant'] The type of padding to add.
     * @param {boolean} [options.center=true] Whether to center the image.
     * @param {number|number[]} [options.constant_values=0] The constant value to use for padding.
     * @returns {[Float32Array, number[]]} The padded pixel data and image dimensions.
     */
    pad_image(pixelData, imgDims, padSize, {
        mode = 'constant',
        center = true,
        constant_values = 0,
    } = {}) {
        return super.pad_image(pixelData, imgDims, padSize, {
            mode,
            center,
            constant_values,
        });
    }
}

