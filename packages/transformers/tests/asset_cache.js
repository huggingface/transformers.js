import { RawImage } from "../src/transformers.js";

const BASE_URL = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/";
const TEST_IMAGES = Object.freeze({
  white_image: BASE_URL + "white-image.png",
  blue_image: BASE_URL + "blue-image.png",
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
  corgi: BASE_URL + "corgi.jpg",
  man_on_car: BASE_URL + "young-man-standing-and-leaning-on-car.jpg",
  portrait_of_woman: BASE_URL + "portrait-of-woman_small.jpg",
});

const TEST_AUDIOS = {
  mlk: BASE_URL + "mlk.npy",
};

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

/** @type {Map<string, any>} */
const AUDIO_CACHE = new Map();
const load_audio = async (url) => {
  const cached = AUDIO_CACHE.get(url);
  if (cached) {
    return cached;
  }
  const buffer = await (await fetch(url)).arrayBuffer();
  const audio = Float32Array.from(new Float64Array(buffer));
  AUDIO_CACHE.set(url, audio);
  return audio;
};

/**
 * Load a cached image.
 * @param {keyof typeof TEST_IMAGES} name The name of the image to load.
 * @returns {Promise<RawImage>} The loaded image.
 */
export const load_cached_image = (name) => load_image(TEST_IMAGES[name]);

/**
 * Load a cached audio.
 * @param {keyof typeof TEST_AUDIOS} name The name of the audio to load.
 * @returns {Promise<Float32Array>} The loaded audio.
 */
export const load_cached_audio = (name) => load_audio(TEST_AUDIOS[name]);
