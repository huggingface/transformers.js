import { RawImage } from "../src/transformers.js";

const BASE_URL = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/";
const TEST_IMAGES = Object.freeze({
  white_image: BASE_URL + "white-image.png",
  pattern_3x3: BASE_URL + "pattern_3x3.png",
  pattern_3x5: BASE_URL + "pattern_3x5.png",
  checkerboard_8x8: BASE_URL + "checkerboard_8x8.png",
  checkerboard_64x32: BASE_URL + "checkerboard_64x32.png",
  gradient_1280x640: BASE_URL + "gradient_1280x640.png",
  receipt: BASE_URL + "receipt.png",
  tiger: BASE_URL + "tiger.jpg",
  paper: BASE_URL + "nougat_paper.png",
  cats: BASE_URL + "cats.jpg",

  // grayscale image
  skateboard: "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/ml-web-games/skateboard.png",

  vitmatte_image: BASE_URL + "vitmatte_image.png",
  vitmatte_trimap: BASE_URL + "vitmatte_trimap.png",

  beetle: BASE_URL + "beetle.png",
  book_cover: BASE_URL + "book-cover.png",
});

/** @type {Map<string, RawImage>} */
const IMAGE_CACHE = new Map();
const load_image = async (url) => {
  const cached = IMAGE_CACHE.get(url);
  if (cached) {
    return cached;
  }
  const image = await RawImage.fromURL(url);
  IMAGE_CACHE.set(url, image);
  return image;
};

/**
 * Load a cached image.
 * @param {keyof typeof TEST_IMAGES} name The name of the image to load.
 * @returns {Promise<RawImage>} The loaded image.
 */
export const load_cached_image = (name) => load_image(TEST_IMAGES[name]);
