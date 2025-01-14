/**
 * @file Helper module for image processing.
 *
 * These functions and classes are only used internally,
 * meaning an end-user shouldn't need to access anything here.
 *
 * @module utils/image
 */

import { getFile } from "./hub.js";
import { env, apis } from "../env.js";
import { Tensor } from "./tensor.js";

// Will be empty (or not used) if running in browser or web-worker
import sharp from "sharp";

type CreateCanvasFunction = (
  width?: number,
  height?: number,
) => OffscreenCanvas;
type LoadImageFunction = (input: sharp.Sharp) => Promise<RawImage>;
type LoadBrowserImageFunction = (
  input: ImageBitmapSource,
) => Promise<ImageBitmap>;
type PadInput = [number, number, number, number];
type CropInput = [number, number, number, number];

let createCanvasFunction: CreateCanvasFunction | undefined;

const IS_BROWSER_OR_WEBWORKER = apis.IS_BROWSER_ENV || apis.IS_WEBWORKER_ENV;

// Function to Get Browser Image Loader
function getBrowserImageLoader(): {
  loadBrowserImageFunction: LoadBrowserImageFunction;
  createCanvasFunction: CreateCanvasFunction;
} {
  try {
    if (!apis.IS_BROWSER_ENV && !apis.IS_WEBWORKER_ENV) {
      throw new Error(
        "This function can only be used in a browser or web worker environment.",
      );
    }

    const createCanvasFunction: CreateCanvasFunction = (
      width?: number,
      height?: number,
    ) => {
      if (!self.OffscreenCanvas) {
        throw new Error("OffscreenCanvas not supported by this browser.");
      }
      return new self.OffscreenCanvas(width!, height!);
    };

    const loadBrowserImageFunction: LoadBrowserImageFunction = async (
      input: ImageBitmapSource,
    ): Promise<ImageBitmap> => {
      return await self.createImageBitmap(input);
    };

    return { loadBrowserImageFunction, createCanvasFunction };
  } catch (error) {
    console.error("Error in getBrowswerImageLoader:", error);
    throw error;
  }
}

// Function to Get Node.js Image Loader
function getNodeImageLoader(): LoadImageFunction {
  try {
    if (!sharp) {
      throw new Error("Sharp library is not available in this environment.");
    }

    const loadImageFunction: LoadImageFunction = async (
      input: sharp.Sharp,
    ): Promise<RawImage> => {
      const metadata = await input.metadata();
      const rawChannels = metadata.channels;

      const { data, info } = await input
        .rotate()
        .raw()
        .toBuffer({ resolveWithObject: true });

      const newImage = new RawImage(
        new Uint8ClampedArray(data),
        info.width,
        info.height,
        info.channels,
      );

      if (rawChannels !== undefined && rawChannels !== info.channels) {
        newImage.convert(rawChannels);
      }

      return newImage;
    };

    return loadImageFunction;
  } catch (error) {
    console.error("Error in getNodeImageLoader:", error);
    throw error;
  }
}

// Defined here: https://github.com/python-pillow/Pillow/blob/a405e8406b83f8bfb8916e93971edc7407b8b1ff/src/libImaging/Imaging.h#L262-L268
const RESAMPLING_MAPPING = Object.freeze({
  0: "nearest",
  1: "lanczos",
  2: "bilinear",
  3: "bicubic",
  4: "box",
  5: "hamming",
} as const);

type ResamplingKeys = keyof typeof RESAMPLING_MAPPING;
type ResamplingValues = (typeof RESAMPLING_MAPPING)[ResamplingKeys];

/**
 * Mapping from file extensions to MIME types.
 */
const CONTENT_TYPE_OBJ = Object.freeze({
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  gif: "image/gif",
} as const);

type ContentTypeKeys = keyof typeof CONTENT_TYPE_OBJ;
type ContentTypeValues = (typeof CONTENT_TYPE_OBJ)[ContentTypeKeys];

/**
 * Create a new `RawImage` object.
 * @param data The pixel data as a buffer.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param channels The number of channels.
 */
export class RawImage {
  data: Uint8ClampedArray | Uint8Array;
  width: number;
  height: number;
  channels: 1 | 2 | 3 | 4;
  static loadImageFunction?: LoadImageFunction;
  static createCanvasFunction?: CreateCanvasFunction;
  static loadBrowswerImageFunction?: LoadBrowserImageFunction;

  constructor(
    data: Uint8ClampedArray,
    width: number,
    height: number,
    channels: 1 | 2 | 3 | 4,
  ) {
    this.data = data;
    this.width = width;
    this.height = height;
    this.channels = channels;
    RawImage.loadImageFunction = getNodeImageLoader();
    const { loadBrowserImageFunction, createCanvasFunction } =
      getBrowserImageLoader();
    RawImage.createCanvasFunction = createCanvasFunction;
    RawImage.loadBrowswerImageFunction = loadBrowserImageFunction;
  }

  /**
   * Returns the size of the image (width, height).
   * @returns The size of the image (width, height).
   */
  get size(): [number, number] {
    return [this.width, this.height];
  }

  /**
   * Helper method for reading an image from a variety of input types.
   * @param {RawImage|string|URL} input
   * @returns The image object.
   *
   * **Example:** Read image from a URL.
   * ```js
   * let image = await RawImage.read('https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/football-match.jpg');
   * // RawImage {
   * //   "data": Uint8ClampedArray [ 25, 25, 25, 19, 19, 19, ... ],
   * //   "width": 800,
   * //   "height": 533,
   * //   "channels": 3
   * // }
   * ```
   */
  static async read(input: RawImage | string | URL): Promise<RawImage> {
    try {
      if (input instanceof RawImage) {
        return input;
      } else if (typeof input === "string" || input instanceof URL) {
        return await RawImage.fromURL(input);
      } else {
        throw new Error(`Unsupported input type: ${typeof input}`);
      }
    } catch (error) {
      console.error("Error in RawImage.read()", error);
      throw error;
    }
  }

  /**
   * Read an image from a canvas.
   * @param {HTMLCanvasElement|OffscreenCanvas} canvas The canvas to read the image from.
   * @returns {RawImage} The image object.
   */
  static fromCanvas(canvas: HTMLCanvasElement | OffscreenCanvas): RawImage {
    try {
      if (!IS_BROWSER_OR_WEBWORKER) {
        throw new Error(
          "fromCanvas() is only supported in browser environments.",
        );
      }

      const ctx = canvas.getContext("2d") as OffscreenCanvasRenderingContext2D;
      if (!ctx) throw new Error("Unable to get canvas rendering context");

      const d = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = d.data;
      // 4 channels for RAW
      return new RawImage(data, canvas.width, canvas.height, 4);
    } catch (error) {
      console.error("Error in RawImage.read()", error);
      throw error;
    }
  }

  /**
   * Read an image from a URL or file path.
   * @param {string | URL}url The URL or file path to read the image from.
   * @returns {Promise<RawImage>} The image object.
   */
  static async fromURL(url: string | URL): Promise<RawImage> {
    try {
      const response = await getFile(url);
      if (!response || response.status !== 200) {
        throw new Error(
          `Unable to read image from "${url}" (${response?.status} ${response?.statusText})`,
        );
      }
      const blob = await response.blob();
      return this.fromBlob(blob);
    } catch (error) {
      console.error("Error in RawImage.fromURL()", error);
      throw error;
    }
  }

  /**
   * Helper method to create a new Image from a blob.
   * @param {Blob} blob The blob to read the image from.
   * @returns {Promise<RawImage>} The image object.
   */
  static async fromBlob(blob: Blob): Promise<RawImage> {
    try {
      if (IS_BROWSER_OR_WEBWORKER) {
        if (!RawImage.loadBrowswerImageFunction || !createCanvasFunction)
          throw new Error("load image function null");
        // Running in environment with canvas
        const img = await RawImage.loadBrowswerImageFunction(blob);

        const ctx = createCanvasFunction(img.width, img.height).getContext(
          "2d",
        );

        if (!ctx) throw new Error("ctx is null");

        // Draw image to context
        ctx.drawImage(img, 0, 0);

        return new RawImage(
          ctx.getImageData(0, 0, img.width, img.height).data,
          img.width,
          img.height,
          4,
        );
      }

      if (!this.loadImageFunction) throw new Error("loadImageFunction is null");
      // Use sharp.js to read (and possible resize) the image.

      const img = sharp(await blob.arrayBuffer());

      return await this.loadImageFunction(img);
    } catch (error) {
      console.error("Error in RawImage.fromBlob()", error);
      throw error;
    }
  }

  /**
   * Helper method to create a new Image from a tensor
   * @param {Tensor} tensor
   */
  fromTensor(tensor: Tensor, channel_format = "CHW"): RawImage {
    try {
      if (tensor.dims.length !== 3) {
        throw new Error(
          `Tensor should have 3 dimensions, but has ${tensor.dims.length} dimensions.`,
        );
      }

      if (channel_format === "CHW") {
        tensor = tensor.transpose(1, 2, 0);
      } else if (channel_format === "HWC") {
        // Do nothing
      } else {
        throw new Error(`Unsupported channel format: ${channel_format}`);
      }
      if (!(tensor.data instanceof Uint8ClampedArray)) {
        throw new Error(`Unsupported tensor type: ${tensor.type}`);
      }
      switch (tensor.dims[2]) {
        case 1:
        case 2:
        case 3:
        case 4:
          return new RawImage(
            tensor.data,
            tensor.dims[1], // height
            tensor.dims[0], // width
            tensor.dims[2],
          );
        default:
          throw new Error(`Unsupported number of channels: ${tensor.dims[2]}`);
      }
    } catch (error) {
      console.error("Error in RawImage.fromTensor()", error);
      throw error;
    }
  }

  /**
   * Convert the image to grayscale format.
   * @returns {RawImage} `this` to support chaining.
   */
  grayscale(): RawImage {
    try {
      // If image is already grayscale, return
      if (this.channels === 1) {
        return this;
      }

      const newData = new Uint8ClampedArray(this.width * this.height * 1);
      switch (this.channels) {
        case 3: // rgb to grayscale
        case 4: // rgba to grayscale
          for (
            let i = 0, offset = 0;
            i < this.data.length;
            i += this.channels
          ) {
            const red = this.data[i];
            const green = this.data[i + 1];
            const blue = this.data[i + 2];

            newData[offset++] = Math.round(
              0.2989 * red + 0.587 * green + 0.114 * blue,
            );
          }
          break;
        default:
          throw new Error(
            `Conversion failed due to unsupported number of channels: ${this.channels}`,
          );
      }
      return this._update(newData, this.width, this.height, 1);
    } catch (error) {
      console.error("Error in RawImage.grayscale()", error);
      throw error;
    }
  }

  /**
   * Convert the image to RGB format.
   * @returns {RawImage} `this` to support chaining.
   */
  rgb(): RawImage {
    try {
      if (this.channels === 3) {
        return this;
      }

      const newData = new Uint8ClampedArray(this.width * this.height * 3);

      switch (this.channels) {
        case 1: // grayscale to rgb
          for (let i = 0, offset = 0; i < this.data.length; ++i) {
            newData[offset++] = this.data[i];
            newData[offset++] = this.data[i];
            newData[offset++] = this.data[i];
          }
          break;
        case 4: // rgba to rgb
          for (let i = 0, offset = 0; i < this.data.length; i += 4) {
            newData[offset++] = this.data[i];
            newData[offset++] = this.data[i + 1];
            newData[offset++] = this.data[i + 2];
          }
          break;
        default:
          throw new Error(
            `Conversion failed due to unsupported number of channels: ${this.channels}`,
          );
      }
      return this._update(newData, this.width, this.height, 3);
    } catch (error) {
      console.error("Error in RawImage.rgb()", error);
      throw error;
    }
  }

  /**
   * Convert the image to RGBA format.
   * @returns {RawImage} `this` to support chaining.
   */
  rgba(): RawImage {
    try {
      // If image is already RGBA, then return
      if (this.channels === 4) {
        return this;
      }

      const newData = new Uint8ClampedArray(this.width * this.height * 4);

      switch (this.channels) {
        case 1: // grayscale to rgba
          for (let i = 0, offset = 0; i < this.data.length; ++i) {
            newData[offset++] = this.data[i];
            newData[offset++] = this.data[i];
            newData[offset++] = this.data[i];
            newData[offset++] = 255;
          }
          break;
        case 3: // rgb to rgba
          for (let i = 0, offset = 0; i < this.data.length; i += 3) {
            newData[offset++] = this.data[i];
            newData[offset++] = this.data[i + 1];
            newData[offset++] = this.data[i + 2];
            newData[offset++] = 255;
          }
          break;
        default:
          throw new Error(
            `Conversion failed due to unsupported number of channels: ${this.channels}`,
          );
      }

      return this._update(newData, this.width, this.height, 4);
    } catch (error) {
      console.error("Error in RawImage.rgba()", error);
      throw error;
    }
  }

  /**
   * Apply an alpha mask to the image. Operates in place.
   * @param {RawImage} mask The mask to apply. It should have a single channel.
   * @returns {RawImage} The masked image.
   * @throws {Error} If the mask is not the same size as the image.
   * @throws {Error} If the image does not have 4 channels.
   * @throws {Error} If the mask is not a single channel.
   */
  putAlpha(mask: RawImage): RawImage {
    try {
      if (mask.width !== this.width || mask.height !== this.height) {
        throw new Error(
          `Expected mask size to be ${this.width}x${this.height}, but got ${mask.width}x${mask.height}`,
        );
      }
      if (mask.channels !== 1) {
        throw new Error(
          `Expected mask to have 1 channel, but got ${mask.channels}`,
        );
      }

      const this_data = this.data;
      const mask_data = mask.data;
      const num_pixels = this.width * this.height;
      if (this.channels === 3) {
        // Convert to RGBA and simultaneously apply mask to alpha channel
        const newData = new Uint8ClampedArray(num_pixels * 4);
        for (let i = 0, in_offset = 0, out_offset = 0; i < num_pixels; ++i) {
          newData[out_offset++] = this_data[in_offset++];
          newData[out_offset++] = this_data[in_offset++];
          newData[out_offset++] = this_data[in_offset++];
          newData[out_offset++] = mask_data[i];
        }
        return this._update(newData, this.width, this.height, 4);
      } else if (this.channels === 4) {
        // Apply mask to alpha channel in place
        for (let i = 0; i < num_pixels; ++i) {
          this_data[4 * i + 3] = mask_data[i];
        }
        return this;
      }
      throw new Error(
        `Expected image to have 3 or 4 channels, but got ${this.channels}`,
      );
    } catch (error) {
      console.error("Error in RawImage.putAlpha()", error);
      throw error;
    }
  }

  /**
   * Resize the image to the given dimensions. This method uses the canvas API to perform the resizing.
   * @param {number} width The width of the new image. `null` or `-1` will preserve the aspect ratio.
   * @param {number} height The height of the new image. `null` or `-1` will preserve the aspect ratio.
   * @param {ResamplingKeys} resample The resampling method to use. Defaults to 2, which is 'bilinear'
   * @returns `this` to support chaining.
   */
  async resize(
    width?: number,
    height?: number,
    { resample }: { resample: ResamplingKeys } = { resample: 2 },
  ): Promise<RawImage> {
    try {
      // Do nothing if the image already has the desired size
      if (this.width === width && this.height === height) {
        return this;
      }

      // Resolve the resampling mapping
      let resampleMethod = RESAMPLING_MAPPING[resample] as ResamplingValues;

      // Calculate width / height to maintain aspect ratio, in the event that
      // the user passed a null value in.
      // This allows users to pass in something like `resize(320, null)` to
      // resize to 320 width, but maintain aspect ratio.

      if (!width && !height) {
        return this;
      } else if (!width && height) {
        // If height is present and width is null
        width = ((height / this.height) * this.width) as number;
      } else if (!height && width) {
        // If width is present and height is null
        height = (width / this.width) * this.height;
      }

      if (IS_BROWSER_OR_WEBWORKER) {
        if (!RawImage.createCanvasFunction)
          throw new Error("Create canvas function null");

        // Store number of channels before resizing
        const numChannels = this.channels;

        // Create canvas object for this image
        const canvas = this.toCanvas();

        // Actually perform resizing using the canvas API
        const ctx = RawImage.createCanvasFunction(width, height).getContext(
          "2d",
        );

        if (!ctx) throw new Error("ctx is null");

        switch (resampleMethod) {
          case "bilinear":
            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = "high";
            break;
          case "nearest":
          case "box":
            ctx.imageSmoothingEnabled = false;
            break;
          case "lanczos":
          case "bicubic":
          case "hamming":
          // Not natively supported
        }

        // Draw image to context, resizing in the process
        ctx.drawImage(canvas, 0, 0, width!, height!);

        // Create image from the resized data
        const resizedImage = new RawImage(
          ctx.getImageData(0, 0, width!, height!).data,
          width!,
          height!,
          4,
        );

        // Convert back so that image has the same number of channels as before
        return resizedImage.convert(numChannels);
      } else {
        if (!RawImage.loadImageFunction)
          throw new Error("loadImageFunction is null");
        // Create sharp image from raw data, and resize
        let img = this.toSharp();

        switch (resampleMethod) {
          case "box":
          case "hamming":
            if (resampleMethod === "box" || resampleMethod === "hamming") {
              console.warn(
                `Resampling method ${resampleMethod} is not yet supported. Using bilinear instead.`,
              );
              resampleMethod = "bilinear";
            }

          case "nearest":
          case "bilinear":
          case "bicubic":
            // Perform resizing using affine transform.
            // This matches how the python Pillow library does it.
            img = img.affine(
              [width! / this.width, 0, 0, height! / this.height],
              {
                interpolator: resampleMethod,
              },
            );
            break;

          case "lanczos":
            // https://github.com/python-pillow/Pillow/discussions/5519
            // https://github.com/lovell/sharp/blob/main/docs/api-resize.md
            img = img.resize({
              width,
              height,
              fit: "fill",
              kernel: "lanczos3", // PIL Lanczos uses a kernel size of 3
            });
            break;

          default:
            throw new Error(
              `Resampling method ${resampleMethod} is not supported.`,
            );
        }

        return await RawImage.loadImageFunction(img);
      }
    } catch (error) {
      console.error("Error in RawImage.resize()", error);
      throw error;
    }
  }

  /**
   * Function to add padding to an image
   * @param {[number, number, number, number]} padInput [left, right, top bottom]
   * @returns `this` to support chaining
   */
  async pad([left, right, top, bottom]: [
    number,
    number,
    number,
    number,
  ]): Promise<RawImage> {
    try {
      left = Math.max(left, 0);
      right = Math.max(right, 0);
      top = Math.max(top, 0);
      bottom = Math.max(bottom, 0);

      if (left === 0 && right === 0 && top === 0 && bottom === 0) {
        // No padding needed
        return this;
      }

      if (IS_BROWSER_OR_WEBWORKER) {
        if (!RawImage.createCanvasFunction)
          throw new Error("createCanvasFunction is null");
        // Store number of channels before padding
        const numChannels = this.channels;

        // Create canvas object for this image
        const canvas = this.toCanvas();

        const newWidth = this.width + left + right;
        const newHeight = this.height + top + bottom;

        // Create a new canvas of the desired size.
        const ctx = RawImage.createCanvasFunction(
          newWidth,
          newHeight,
        ).getContext("2d");
        if (!ctx) throw new Error("ctx is null");

        // Draw image to context, padding in the process
        ctx.drawImage(
          canvas,
          0,
          0,
          this.width,
          this.height,
          left,
          top,
          this.width,
          this.height,
        );

        // Create image from the padded data
        const paddedImage = new RawImage(
          ctx.getImageData(0, 0, newWidth, newHeight).data,
          newWidth,
          newHeight,
          4,
        );

        // Convert back so that image has the same number of channels as before
        return paddedImage.convert(numChannels);
      } else {
        if (!RawImage.loadImageFunction)
          throw new Error("loadImageFunction is null");
        const img = this.toSharp().extend({ left, right, top, bottom });
        return await RawImage.loadImageFunction(img);
      }
    } catch (error) {
      console.error("Error in RawImage.pad()", error);
      throw error;
    }
  }

  /**
   * Crop an image to specified output
   * @param {[number, number, number, number]} params [x_min, y_min, x_max, y_max]
   * @returns `this` to support chaining
   */
  async crop([x_min, y_min, x_max, y_max]: [
    number,
    number,
    number,
    number,
  ]): Promise<RawImage> {
    try {
      // Ensure crop bounds are within the image
      x_min = Math.max(x_min, 0);
      y_min = Math.max(y_min, 0);
      x_max = Math.min(x_max, this.width - 1);
      y_max = Math.min(y_max, this.height - 1);

      // Do nothing if the crop is the entire image
      if (
        x_min === 0 &&
        y_min === 0 &&
        x_max === this.width - 1 &&
        y_max === this.height - 1
      ) {
        return this;
      }

      const crop_width = x_max - x_min + 1;
      const crop_height = y_max - y_min + 1;

      if (IS_BROWSER_OR_WEBWORKER) {
        if (!RawImage.createCanvasFunction)
          throw new Error("createCanvasFunction is null");
        // Store number of channels before resizing
        const numChannels = this.channels;

        // Create canvas object for this image
        const canvas = this.toCanvas();

        // Create a new canvas of the desired size. This is needed since if the
        // image is too small, we need to pad it with black pixels.
        const ctx = RawImage.createCanvasFunction(
          crop_width,
          crop_height,
        ).getContext("2d");

        if (!ctx) throw new Error("ctx is null");

        // Draw image to context, cropping in the process
        ctx.drawImage(
          canvas,
          x_min,
          y_min,
          crop_width,
          crop_height,
          0,
          0,
          crop_width,
          crop_height,
        );

        // Create image from the resized data
        const resizedImage = new RawImage(
          ctx.getImageData(0, 0, crop_width, crop_height).data,
          crop_width,
          crop_height,
          4,
        );

        // Convert back so that image has the same number of channels as before
        return resizedImage.convert(numChannels);
      }
      if (!RawImage.loadImageFunction)
        throw new Error("loadImageFunction is null");
      // Create sharp image from raw data
      const img = this.toSharp().extract({
        left: x_min,
        top: y_min,
        width: crop_width,
        height: crop_height,
      });

      return await RawImage.loadImageFunction(img);
    } catch (error) {
      console.error("Error in RawImage.crop()", error);
      throw error;
    }
  }

  /**
   * Crop image to desired dimensions
   * @param {number} crop_width
   * @param {number} crop_height
   * @returns `this` to support chaining
   */
  async center_crop(
    crop_width: number,
    crop_height: number,
  ): Promise<RawImage> {
    try {
      // If the image is already the desired size, return it
      if (this.width === crop_width && this.height === crop_height) {
        return this;
      }

      // Determine bounds of the image in the new canvas
      const width_offset = (this.width - crop_width) / 2;
      const height_offset = (this.height - crop_height) / 2;

      if (IS_BROWSER_OR_WEBWORKER) {
        if (!RawImage.createCanvasFunction)
          throw new Error("createCanvasFunction is null");
        // Store number of channels before resizing
        const numChannels = this.channels;

        // Create canvas object for this image
        const canvas = this.toCanvas();

        // Create a new canvas of the desired size. This is needed since if the
        // image is too small, we need to pad it with black pixels.
        const ctx = RawImage.createCanvasFunction(
          crop_width,
          crop_height,
        ).getContext("2d");
        if (!ctx) throw new Error("ctx is null");

        let sourceX = 0;
        let sourceY = 0;
        let destX = 0;
        let destY = 0;

        if (width_offset >= 0) {
          sourceX = width_offset;
        } else {
          destX = -width_offset;
        }

        if (height_offset >= 0) {
          sourceY = height_offset;
        } else {
          destY = -height_offset;
        }

        // Draw image to context, cropping in the process
        ctx.drawImage(
          canvas,
          sourceX,
          sourceY,
          crop_width,
          crop_height,
          destX,
          destY,
          crop_width,
          crop_height,
        );

        // Create image from the resized data
        const resizedImage = new RawImage(
          ctx.getImageData(0, 0, crop_width, crop_height).data,
          crop_width,
          crop_height,
          4,
        );

        // Convert back so that image has the same number of channels as before
        return resizedImage.convert(numChannels);
      }

      if (!RawImage.loadImageFunction)
        throw new Error("loadImageFunction is null");
      // Create sharp image from raw data
      let img = this.toSharp();

      if (width_offset >= 0 && height_offset >= 0) {
        // Cropped image lies entirely within the original image
        img = img.extract({
          left: Math.floor(width_offset),
          top: Math.floor(height_offset),
          width: crop_width,
          height: crop_height,
        });
      } else if (width_offset <= 0 && height_offset <= 0) {
        // Cropped image lies entirely outside the original image,
        // so we add padding
        const top = Math.floor(-height_offset);
        const left = Math.floor(-width_offset);
        img = img.extend({
          top: top,
          left: left,

          // Ensures the resulting image has the desired dimensions
          right: crop_width - this.width - left,
          bottom: crop_height - this.height - top,
        });
      } else {
        // Cropped image lies partially outside the original image.
        // We first pad, then crop.

        let y_padding = [0, 0];
        let y_extract = 0;
        if (height_offset < 0) {
          y_padding[0] = Math.floor(-height_offset);
          y_padding[1] = crop_height - this.height - y_padding[0];
        } else {
          y_extract = Math.floor(height_offset);
        }

        let x_padding = [0, 0];
        let x_extract = 0;
        if (width_offset < 0) {
          x_padding[0] = Math.floor(-width_offset);
          x_padding[1] = crop_width - this.width - x_padding[0];
        } else {
          x_extract = Math.floor(width_offset);
        }

        img = img
          .extend({
            top: y_padding[0],
            bottom: y_padding[1],
            left: x_padding[0],
            right: x_padding[1],
          })
          .extract({
            left: x_extract,
            top: y_extract,
            width: crop_width,
            height: crop_height,
          });
      }
      return await RawImage.loadImageFunction(img);
    } catch (error) {
      console.error("Error in RawImage.centerCrop()", error);
      throw error;
    }
  }

  /**
   * Output RawImage as a blob
   * @param {ContentTypeValues} type MIME type of the image. Default is `image/png`
   * @param {number} quality A Number between 0 and 1 indicating the image quality to be used when creating images using file formats that support lossy compression (such as `image/jpeg` or `image/webp`).
   * @returns Blob
   */
  async toBlob(
    type: ContentTypeValues = "image/png",
    quality: number = 1,
  ): Promise<Blob> {
    try {
      if (!IS_BROWSER_OR_WEBWORKER) {
        throw new Error(
          "This method is only supported in browser environments.",
        );
      }

      const canvas = this.toCanvas();
      return await canvas.convertToBlob({ type, quality });
    } catch (error) {
      console.error("Error in RawImage.toBlob()", error);
      throw error;
    }
  }

  /**
   * Output RawImage to Tensor
   * @param channel_format Default value is "CHW"
   * @returns Tensor
   */
  toTensor(channel_format = "CHW"): Tensor {
    try {
      let tensor = new Tensor("uint8", new Uint8Array(this.data), [
        this.height,
        this.width,
        this.channels,
      ]);

      if (channel_format === "HWC") {
        // Do nothing
      } else if (channel_format === "CHW") {
        // hwc -> chw
        tensor = tensor.permute(2, 0, 1);
      } else {
        throw new Error(`Unsupported channel format: ${channel_format}`);
      }
      return tensor;
    } catch (error) {
      console.error("Error in RawImage.toTensor()", error);
      throw error;
    }
  }

  /**
   * Output RawImage to Canvas
   *
   * ** Only supported in browser environments **
   * @returns OffscreenCanvas
   */
  toCanvas(): OffscreenCanvas {
    try {
      if (!IS_BROWSER_OR_WEBWORKER) {
        throw new Error(
          "toCanvas() is only supported in browser environments.",
        );
      }
      if (!RawImage.createCanvasFunction)
        throw new Error("createCanvasFn is null");

      // Clone, and convert data to RGBA before drawing to canvas.
      // This is because the canvas API only supports RGBA
      const cloned = this.clone().rgba();

      // Create canvas object for the cloned image
      const clonedCanvas = RawImage.createCanvasFunction(
        cloned.width,
        cloned.height,
      );

      let clonedData: Uint8ClampedArray;

      // ImageData class wants Uint8ClampedArray, so we convert to ensure
      if (cloned.data instanceof Uint8Array) {
        // Convert Uint8Array to Uint8ClampedArray
        clonedData = new Uint8ClampedArray(cloned.data.buffer);
      } else {
        // Assume cloned.data is already a Uint8ClampedArray
        clonedData = cloned.data as Uint8ClampedArray;
      }

      // Draw image to context
      const data = new ImageData(clonedData, cloned.width, cloned.height);
      const ctx = clonedCanvas.getContext("2d");
      if (!ctx) throw new Error("ctx is null");

      ctx.putImageData(data, 0, 0);

      return clonedCanvas;
    } catch (error) {
      console.error("Error in RawImage.toCanvas()", error);
      throw error;
    }
  }

  /**
   * Split this image into individual bands. This method returns an array of individual image bands from an image.
   * For example, splitting an "RGB" image creates three new images each containing a copy of one of the original bands (red, green, blue).
   *
   * Inspired by PIL's `Image.split()` [function](https://pillow.readthedocs.io/en/latest/reference/Image.html#PIL.Image.Image.split).
   * @returns {RawImage[]} An array containing bands as RawImage.
   */
  split(): RawImage[] {
    try {
      const { data, width, height, channels } = this;

      const per_channel_length = data.length / channels;

      // Pre-allocate buffers for each channel
      const split_data = Array.from(
        { length: channels },
        () => new Uint8ClampedArray(per_channel_length),
      );

      // Write pixel data
      for (let i = 0; i < per_channel_length; ++i) {
        const data_offset = channels * i;
        for (let j = 0; j < channels; ++j) {
          split_data[j][i] = data[data_offset + j];
        }
      }
      return split_data.map((data) => new RawImage(data, width, height, 1));
    } catch (error) {
      console.error("Error in RawImage.split()", error);
      throw error;
    }
  }

  /**
   * Helper method to update the image data.
   * @param {Uint8ClampedArray | Uint8Array} data The new image data.
   * @param {number} width The new width of the image.
   * @param {number} height The new height of the image.
   * @param {1 | 2 | 3 | 4} channels The new number of channels of the image.
   * @private
   */
  private _update(
    data: Uint8ClampedArray | Uint8Array,
    width: number,
    height: number,
    channels?: 1 | 2 | 3 | 4,
  ) {
    this.data = data;
    this.width = width;
    this.height = height;
    if (channels) this.channels = channels;
    return this;
  }

  /**
   * Clone the image
   * @returns {RawImage} The cloned image
   */
  clone(): RawImage {
		// Slice method only exists on clamped arrays
		// Convert to ensure compatibility
    let clonedData: Uint8ClampedArray;

    if (this.data instanceof Uint8Array) {
      // Convert Uint8Array to Uint8ClampedArray
      clonedData = new Uint8ClampedArray(this.data.buffer);
    } else {
      // Assume cloned.data is already a Uint8ClampedArray
      clonedData = this.data as Uint8ClampedArray;
    }

    return new RawImage(
      clonedData.slice(),
      this.width,
      this.height,
      this.channels,
    );
  }

  /**
   * Helper method for converting image to have a certain number of channels
   * @param {number} numChannels The number of channels. Must be 1, 3, or 4.
   * @returns `this` to support chaining.
   */
  convert(numChannels: number): RawImage {
    try {
      if (this.channels === numChannels) return this; // Already correct number of channels

      switch (numChannels) {
        case 1:
          this.grayscale();
          break;
        case 3:
          this.rgb();
          break;
        case 4:
          this.rgba();
          break;
        default:
          throw new Error(
            `Conversion failed due to unsupported number of channels: ${this.channels}`,
          );
      }
      return this;
    } catch (error) {
      console.error("Error in RawImage.convert()", error);
      throw error;
    }
  }

  /**
   * Save the image to the given path.
   * @param {string} path The path to save the image to.
	 * @returns {sharp.OutputInfo | undefined} If in node, returns output information. 
	 * 
	 * Else if in browser, returns undefined.
   */
  async save(path: string): Promise<sharp.OutputInfo | undefined> {
    try {
      if (IS_BROWSER_OR_WEBWORKER) {
        if (apis.IS_WEBWORKER_ENV) {
          throw new Error("Unable to save an image from a Web Worker.");
        }

        const extension = path.split(".").pop()?.toLowerCase();

        const mime =
          extension && extension in CONTENT_TYPE_OBJ
            ? CONTENT_TYPE_OBJ[extension as ContentTypeKeys]
            : "image/png";

        // Convert image to Blob
        const blob = await this.toBlob(mime);

        // Convert the canvas content to a data URL
        const dataURL = URL.createObjectURL(blob);

        // Create an anchor element with the data URL as the href attribute
        const downloadLink = document.createElement("a");
        downloadLink.href = dataURL;

        // Set the download attribute to specify the desired filename for the downloaded image
        downloadLink.download = path;

        // Trigger the download
        downloadLink.click();

        // Clean up: remove the anchor element from the DOM
        downloadLink.remove();

        return;
      }

      if (!env.useFS)
        throw new Error(
          "Unable to save the image because filesystem is disabled in this environment.",
        );

      const img = this.toSharp();
      return await img.toFile(path);
    } catch (error) {
      console.error("Error in RawImage.save()", error);
      throw error;
    }
  }

  /**
   * Transforms RawImage to sharp object for chaining
   * @returns {sharp.Sharp} Sharp object
   */
  toSharp(): sharp.Sharp {
    try {
      if (IS_BROWSER_OR_WEBWORKER) {
        throw new Error(
          "toSharp() is only supported in server-side environments.",
        );
      }

      return sharp(this.data, {
        raw: {
          width: this.width,
          height: this.height,
          channels: this.channels,
        },
      });
    } catch (error) {
      console.error("Error in RawImage.toSharp()", error);
      throw error;
    }
  }
}

/**
 * Helper function to load an image from a URL, path, etc.
 */
export const load_image = RawImage.read.bind(RawImage);
